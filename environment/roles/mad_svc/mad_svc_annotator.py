import json
import os


from environment.agents.base import BaseAgent
import mido

from environment.communication.message import Message


def note_to_name(note_number):
    """将MIDI音符号转换为音名"""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = note_number // 12 - 1
    note = notes[note_number % 12]
    return f"{note}{octave}"


def get_tempo_changes(mid):
    """获取所有tempo变化点"""
    tempo_changes = []
    # 默认tempo
    default_tempo = 500000
    for track in mid.tracks:
        track_time = 0
        for msg in track:
            track_time += msg.time
            if msg.type == 'set_tempo':
                tempo_changes.append({
                    'time': track_time,
                    'tempo': msg.tempo
                })
    # 按时间排序
    tempo_changes.sort(key=lambda x: x['time'])
    # 如果一开始没有tempo，就插入默认tempo
    if not tempo_changes or tempo_changes[0]['time'] > 0:
        tempo_changes.insert(0, {'time': 0, 'tempo': default_tempo})
    return tempo_changes


def ticks_to_seconds(start_ticks, duration_ticks, tempo_changes, ticks_per_beat):
    """将tick转换为秒数，考虑tempo变化"""
    end_ticks = start_ticks + duration_ticks
    duration_seconds = 0
    current_ticks = start_ticks

    for i in range(len(tempo_changes)):
        next_change_ticks = tempo_changes[i + 1]['time'] if i + 1 < len(tempo_changes) else float('inf')
        current_tempo = tempo_changes[i]['tempo']

        if current_ticks >= end_ticks:
            break

        segment_end_ticks = min(end_ticks, next_change_ticks)
        segment_duration_ticks = segment_end_ticks - current_ticks

        # 转换这段时间为秒数
        duration_seconds += (segment_duration_ticks * current_tempo) / (ticks_per_beat * 1000000)
        current_ticks = segment_end_ticks

    return duration_seconds


def count_actual_notes(notes_str):
    """计算实际的音符数量（不包括休止符）"""
    notes = notes_str.split(' | ')
    return sum(1 for note in notes if note != 'rest')


def analyze_midi(midi_path, lyrics, output_path):
    try:
        mid = mido.MidiFile(midi_path)
        track_results = {}
        tempo_changes = get_tempo_changes(mid)
        output_data = {}

        # 打印所有tempo信息
        print("\nTempo changes:")
        for tc in tempo_changes:
            bpm = 60000000 / tc['tempo']
            print(f"At tick {tc['time']}: {bpm:.2f} BPM")

        # 提取有音符的轨道
        for track_idx, track in enumerate(mid.tracks):
            track_name = track.name if track.name else f"Track {track_idx}"
            notes_list = []
            duration_list = []

            notes = []
            current_time = 0
            current_notes = {}
            last_note_end = 0  # 记录上一个音符结束的时间

            for msg in track:
                current_time += msg.time

                # 如果是note_on且velocity>0，说明新音符开始
                if msg.type == 'note_on' and msg.velocity > 0:
                    # 判断是否出现休止符
                    if current_time - last_note_end > mid.ticks_per_beat / 8:
                        rest_duration = current_time - last_note_end
                        notes.append({
                            'note': 'rest',
                            'start': last_note_end,
                            'duration': rest_duration
                        })
                    current_notes[msg.note] = {
                        'start': current_time,
                        'velocity': msg.velocity
                    }

                # 如果是note_off，或是note_on且velocity=0，则说明音符结束
                elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in current_notes:
                        start_time = current_notes[msg.note]['start']
                        duration = current_time - start_time
                        notes.append({
                            'note': msg.note,
                            'start': start_time,
                            'duration': duration
                        })
                        last_note_end = current_time
                        del current_notes[msg.note]

            # 按开始时间排序，方便后面合并同时发音
            notes.sort(key=lambda x: x['start'])

            # 找出同时发音的音符
            current_time = 0
            current_group = []
            current_duration = 0

            for note in notes:
                # 如果还在同一时间点（同时发音）
                if not current_group or abs(note['start'] - current_time) < 0.01:
                    current_group.append(note)
                    current_duration = max(current_duration, note['duration'])
                else:
                    # 把前一个时间点的音符信息先保存
                    if current_group[0]['note'] == 'rest':
                        notes_list.append('rest')
                    else:
                        note_names = ' '.join([note_to_name(n['note']) for n in current_group])
                        notes_list.append(note_names)

                    duration_seconds = ticks_to_seconds(current_time, current_duration,
                                                        tempo_changes, mid.ticks_per_beat)
                    duration_list.append(f"{duration_seconds:.6f}")

                    # 开始新的时间段
                    current_time = note['start']
                    current_group = [note]
                    current_duration = note['duration']

            # 处理最后一组
            if current_group:
                if current_group[0]['note'] == 'rest':
                    notes_list.append('rest')
                else:
                    note_names = ' '.join([note_to_name(n['note']) for n in current_group])
                    notes_list.append(note_names)

                duration_seconds = ticks_to_seconds(current_time, current_duration,
                                                    tempo_changes, mid.ticks_per_beat)
                duration_list.append(f"{duration_seconds:.6f}")

            # 只保存有音符的轨道
            if notes_list:
                track_results[track_name] = {
                    'notes': ' | '.join(notes_list),
                    'notes_duration': ' | '.join(duration_list)
                }

        print(f"\nMIDI文件共有 {len(mid.tracks)} 个轨道")

        # 处理匹配的轨道并生成JSON
        for track_name, result in track_results.items():
            actual_notes_count = count_actual_notes(result['notes'])
            print(track_name)
            if actual_notes_count == len(lyrics):
                print(f"\n=== {track_name} ===")
                print(f"实际音符数量（不含休止符）: {actual_notes_count}")
                print(f"歌词字数: {len(lyrics)}")

                processed_lyrics = []
                current_lyric_index = 0
                notes_split = result['notes'].split(' | ')

                for note_str in notes_split:
                    if note_str == 'rest':
                        processed_lyrics.append("AP")
                    else:
                        if current_lyric_index < len(lyrics):
                            processed_lyrics.append(lyrics[current_lyric_index])
                            current_lyric_index += 1

                print("\n插入后的歌词：")
                print("".join(processed_lyrics))

                print("\nNotes:")
                print(result['notes'])
                print("\nDurations (seconds):")
                print(result['notes_duration'])

                output_data = {
                    'text': ''.join(processed_lyrics),
                    'notes': result['notes'],
                    'notes_duration': result['notes_duration'],
                    'input_type': 'word'
                }

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)

                return result

    except Exception as e:
        print(f"Error analyzing MIDI file: {str(e)}")
        return None


class MadSVCAnnotator(BaseAgent):
    def __init__(self):
        super().__init__()

    def process_message(self, message):
        midi_path = message.content.get("midi")
        lyrics_path = message.content.get("lyrics")

        if not midi_path.endswith('.mid'):
            return {
                "status": "error",
                "message": "MIDI file must have .mid extension"
            }

        # 判断lyrics path的后缀名是否为txt
        if not lyrics_path.endswith('.txt'):
            return {
                "status": "error",
                "message": "Lyrics file must have .txt extension"
            }

        # 取lyrics_path的文件名，即.txt前面的部分
        name = os.path.splitext(os.path.basename(lyrics_path))[0]
        # 读取歌词文件
        try:
            with open(lyrics_path, 'r', encoding='utf-8') as f:
                lyrics = f.read()
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to read lyrics file: {str(e)}"
            }

        try:
            # 确保输出目录存在
            os.makedirs("dataset/mad_svc/lyrics_annotation", exist_ok=True)
            output_path = "dataset/mad_svc/lyrics_annotation/" + name + ".json"
            json_result = analyze_midi(midi_path, lyrics, output_path)

            return Message(
                content={
                "status": "success",
                "name": f"{name}",
                "output_path": f"{output_path}",
                "result": json_result
            })


        except Exception as e:
            return Message(
                content={
                "status": "error",
                "message": str(e)  # 确保错误信息是字符串
            })
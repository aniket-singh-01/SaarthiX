import os
import json
from pydub import AudioSegment

from environment.agents.base import BaseAgent
from environment.communication.message import Message
from tools.DiffSinger.diff import run_diffsinger


class MadSVCSpliter(BaseAgent):
    def __init__(self):
        super().__init__()

    def process_message(self, message):
        """处理消息，分离长AP并生成音频（带时间补偿）"""
        analysis = message.content.get("analyzer_result").content
        annotator_result = message.content.get("annotator_result").content

        inp = {
            "text": analysis["lyrics"],
            "notes": annotator_result["result"]["notes"],
            "notes_duration": annotator_result["result"]["notes_duration"],
            "input_type": "word"
        }

        text = inp["text"]
        notes_list = inp["notes"].split(" | ")
        notes_duration_str_list = inp["notes_duration"].split(" | ")
        notes_duration_float_list = [float(d) for d in notes_duration_str_list]

        tokens, ap_segments = self._tokenize_text(
            text, notes_list, notes_duration_str_list, notes_duration_float_list
        )

        # 无长AP直接处理
        if not ap_segments:
            return self._process_single_segment(inp, annotator_result)

        # 分割段落
        segments = self._split_segments(tokens, ap_segments)

        # 计算理论时长
        segment_theoretical_durations = [
            sum(float(t["duration_str"]) for t in seg) for seg in segments
        ]

        # 生成各段落音频
        new_name, output_audio_files = self._generate_audio_segments(
            segments, annotator_result
        )

        first_ap_at_start = False
        if ap_segments:
            first_ap_at_start = (ap_segments[0]["position"] == 0)

        return self._merge_with_compensation(
            new_name, output_audio_files, ap_segments,
            segment_theoretical_durations, first_ap_at_start
        )

    def _tokenize_text(self, text, notes_list, dur_str, dur_float):
        """分词处理并记录时间信息"""
        tokens = []
        ap_segments = []
        notes_pos = i = 0

        while i < len(text):
            if i <= len(text) - 2 and text[i:i + 2] == "AP":
                current_dur = dur_float[notes_pos] if notes_pos < len(dur_float) else 0.3
                if current_dur > 0.5:
                    ap_segments.append({
                        "position": len(tokens),
                        "duration": current_dur,
                        "order": len(ap_segments)
                    })
                    notes_pos += 1
                    i += 2
                else:
                    tokens.append(self._create_token(
                        "AP", notes_list, dur_str, notes_pos
                    ))
                    notes_pos += 1
                    i += 2
            else:
                if text[i] != ' ':
                    tokens.append(self._create_token(
                        text[i], notes_list, dur_str, notes_pos
                    ))
                    notes_pos += 1
                i += 1
        return tokens, ap_segments

    def _create_token(self, text, notes, durations, pos):
        """创建token数据"""
        return {
            "text": text,
            "note": notes[pos] if pos < len(notes) else "rest",
            "duration_str": durations[pos] if pos < len(durations) else "0.3"
        }

    def _split_segments(self, tokens, ap_segments):
        """按AP位置分割段落"""
        split_points = sorted([ap["position"] for ap in ap_segments])
        segments = []
        prev = 0
        for point in split_points:
            if point > prev:
                segments.append(tokens[prev:point])
            prev = point
        if prev < len(tokens):
            segments.append(tokens[prev:])
        return segments

    def _generate_audio_segments(self, segments, annotator_result):
        """生成各段落音频并返回路径"""
        new_name = annotator_result['name'] + '_cover'
        output_audio_files = []

        os.makedirs("dataset/mad_svc/lyrics_annotation", exist_ok=True)
        os.makedirs("dataset/mad_svc/cover", exist_ok=True)

        for idx, seg_tokens in enumerate(segments):
            segment = self._create_segment(seg_tokens)
            segment_name = f"{new_name}_part_{idx}"
            inp_path = f'dataset/mad_svc/lyrics_annotation/{segment_name}.json'

            with open(inp_path, 'w', encoding='utf-8') as f:
                json.dump(segment, f, ensure_ascii=False, indent=2)

            run_diffsinger(inp='../../' + inp_path, save_name=segment_name)

            output_file = f"dataset/mad_svc/cover/{segment_name}.wav"
            if os.path.exists(output_file):
                output_audio_files.append(output_file)

        return new_name, output_audio_files

    def _merge_with_compensation(self, name, audio_files, ap_segments,
                                theoretical_durs, first_ap_at_start):
        """动态选择合并逻辑的核心方法"""
        # 测量各段实际时长
        actual_durations = []
        for file in audio_files:
            audio = AudioSegment.from_file(file)
            actual_durations.append(len(audio) / 1000.0)  # 转换为秒

        # 初始化变量
        time_compensation = 0.0
        final_audio = AudioSegment.empty()
        sorted_ap = sorted(ap_segments, key=lambda x: x["order"])
        ap_index = 0  # 当前处理的AP索引

        # 根据第一个AP是否在开头选择逻辑分支
        if first_ap_at_start:
            # 逻辑分支1：AP在开头，先插入静音再添加音频
            for seg_idx in range(len(audio_files)):
                # 插入当前AP对应的静音
                if ap_index < len(sorted_ap):
                    ap = sorted_ap[ap_index]
                    adjusted_duration = ap["duration"] + time_compensation
                    adjusted_duration = max(adjusted_duration, 0)
                    silence = AudioSegment.silent(duration=int(adjusted_duration * 1000))
                    final_audio += silence
                    # 更新补偿值（静音实际插入的调整量）
                    time_compensation -= (adjusted_duration - ap["duration"])
                    ap_index += 1

                # 添加当前段落音频
                with open(audio_files[seg_idx], 'rb') as f:
                    seg_audio = AudioSegment.from_file(f)
                    final_audio += seg_audio

                # 计算当前段落的时间补偿
                theoretical = theoretical_durs[seg_idx]
                actual = actual_durations[seg_idx]
                time_compensation += (theoretical - actual)
        else:
            # 逻辑分支2：AP不在开头，先添加音频再插入静音
            for seg_idx in range(len(audio_files)):
                # 添加当前段落音频
                with open(audio_files[seg_idx], 'rb') as f:
                    seg_audio = AudioSegment.from_file(f)
                    final_audio += seg_audio

                # 计算当前段落的时间补偿
                theoretical = theoretical_durs[seg_idx]
                actual = actual_durations[seg_idx]
                time_compensation += (theoretical - actual)

                # 插入当前AP对应的静音
                if ap_index < len(sorted_ap):
                    ap = sorted_ap[ap_index]
                    adjusted_duration = ap["duration"] + time_compensation
                    adjusted_duration = max(adjusted_duration, 0)
                    silence = AudioSegment.silent(duration=int(adjusted_duration * 1000))
                    final_audio += silence
                    # 更新补偿值
                    time_compensation -= (adjusted_duration - ap["duration"])
                    ap_index += 1

        # 导出最终音频
        final_path = f"dataset/mad_svc/cover/{name}.wav"
        final_audio.export(final_path, format="wav")
        return Message(content={"final_path": final_path})

    def _process_single_segment(self, inp, annotator_result):
        """处理无分割的情况"""
        new_name = annotator_result['name'] + '_cover'
        inp_path = f'dataset/mad_svc/lyrics_annotation/{new_name}.json'
        with open(inp_path, 'w', encoding='utf-8') as f:
            json.dump(inp, f, ensure_ascii=False, indent=2)
        return run_diffsinger(inp='../../' + inp_path, save_name=new_name)

    def _create_segment(self, tokens):
        """创建段落数据"""
        return {
            "text": "".join(t["text"] for t in tokens),
            "notes": " | ".join(t["note"] for t in tokens),
            "notes_duration": " | ".join(t["duration_str"] for t in tokens),
            "input_type": "word"
        }
import json
import os
import subprocess
from pathlib import Path
from datetime import timedelta

from environment.agents.base import BaseAgent


class MadSVCSubtitle(BaseAgent):
    def __init__(self):
        super().__init__()
        self.temp_srt = Path("subs.srt")

    def _format_timestamp(self, seconds):
        """将秒数转换为SRT标准时间格式"""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(int(td.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int(td.microseconds / 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def _generate_srt_from_json(self, json_path):
        """从时间戳JSON生成SRT字幕文件"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not data.get("segments"):
                raise ValueError("JSON文件中缺少segments字段")

            segments = sorted(data["segments"], key=lambda x: x["start"])

            srt_content = []
            for idx, seg in enumerate(segments, 1):
                # 时间戳格式转换
                start = self._format_timestamp(seg["start"])
                end = self._format_timestamp(seg["end"])

                # 字幕文本处理（保留原格式）
                text = seg["text"].replace('\\n', '\n')  # 处理可能存在的换行符

                srt_content.append(
                    f"{idx}\n"
                    f"{start} --> {end}\n"
                    f"{text}\n"
                )

            # 写入临时SRT文件
            with self.temp_srt.open('w', encoding='utf-8') as f:
                f.write('\n'.join(srt_content))

        except json.JSONDecodeError:
            raise ValueError("无效的JSON格式")
        except KeyError as e:
            raise ValueError(f"JSON字段缺失: {str(e)}")

    def _burn_subtitles(self, input_video, output_video):
        """烧录字幕到视频"""
        if not self.temp_srt.exists():
            raise FileNotFoundError("字幕文件未生成")

        cmd = [
            'ffmpeg',
            '-y',
            '-i', str(input_video),
            '-vf', f"subtitles={self.temp_srt.name}:force_style="
                   "'FontName=Microsoft YaHei,"
                   "FontSize=10,"
                   "PrimaryColour=&HFFFFFF,"
                   "OutlineColour=&H000000,"
                   "BorderStyle=3,"
                   "BackColour=&H80000000,"
                   "MarginV=20,"
                   "Outline=0.5'",
            '-c:a', 'copy',
            str(output_video)
        ]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding='utf-8',
                errors='replace',
                cwd=str(self.temp_srt.parent)
            )
            print("字幕烧录成功，输出日志:", result.stdout)
        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg执行失败: {e.stdout}"
            print(error_msg)
            raise RuntimeError(error_msg)

    def calculate_accurate_timestamps(self, annotation_path):
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        text = data['text']
        durations = list(map(float, data['notes_duration'].split(' | ')))

        # 构建时间轴（新增duration验证）
        timeline = []
        i = j = current_time = 0
        while i < len(text) and j < len(durations):
            if i < len(text) - 1 and text[i] == 'A' and text[i + 1] == 'P':
                # 记录AP时间戳（精确到毫秒）
                timeline.append(('AP', current_time, current_time + durations[j]))
                current_time += durations[j]
                j += 1
                i += 2
            else:
                # 字符级时间记录
                char_end = current_time + durations[j]
                timeline.append(('CHAR', text[i], current_time, char_end))
                current_time = char_end  # 关键修正点：持续累加时间
                j += 1
                i += 1

        # 生成段落（新增时间累积逻辑）
        segments = []
        last_ap_end = 0.0
        current_segment = None

        for item in timeline:
            if item[0] == 'AP':
                if current_segment:
                    # 闭合前一段落：end = 当前AP的start时间
                    current_segment['end'] = item[1]
                    segments.append(current_segment)
                    current_segment = None
                last_ap_end = item[2]  # 更新AP结束时间
            else:
                if not current_segment:
                    # 新段落开始：start = 前一个AP的end时间
                    current_segment = {
                        'text': '',
                        'start': last_ap_end,
                        'end': last_ap_end  # 初始化为start时间
                    }
                # 累加字符和时间
                current_segment['text'] += item[1]
                current_segment['end'] = item[3]  # 更新为字符的结束时间

        # 处理最后未闭合的段落
        if current_segment:
            current_segment['end'] = current_time
            segments.append(current_segment)

        # 生成输出（新增时间校验）
        output = {
            "segments": [{
                "text": seg['text'],
                "start": round(seg['start'], 3),
                "end": round(seg['end'], 3)
            } for seg in segments if seg['end'] > seg['start']],  # 过滤无效段落
            "total_duration": round(current_time, 3)
        }

        output_path = Path(annotation_path).parent / "slice_timestamp.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        return output_path  # 返回生成的JSON路径

    def process_message(self, message):
        """完整处理流程"""
        annotation_path = Path(message.content["annotation_path"])
        input_video = Path(message.content["video_path"])
        output_video = Path(message.content["output_path"])
        print(annotation_path)
        with open('dataset/mad_svc/lyrics_annotation/有何不可.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        with open('dataset/mad_svc/script.txt', 'r', encoding='utf-8') as f:
            result = f.read()
        data['text'] = result
        with open('dataset/mad_svc/lyrics_annotation/有何不可_cover.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 1. 生成时间戳JSON
        timestamp_json = self.calculate_accurate_timestamps(annotation_path)

        # 2. 生成SRT字幕
        self._generate_srt_from_json(timestamp_json)

        # 3. 烧录字幕到视频
        self._burn_subtitles(input_video, output_video)

        # 4. 清理临时文件
        self.temp_srt.unlink(missing_ok=True)

        return {"status": "success", "output_path": str(output_video)}
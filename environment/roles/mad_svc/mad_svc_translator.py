import json
import os.path
import time
from pathlib import Path
from environment.agents.base import BaseAgent

class MadSVCTranslator(BaseAgent):
    def __init__(self):
        super().__init__()

    def parse_text_to_segments(self, text, durations):
        """解析文本并生成时间段落，返回 segments 和 AP 时间段"""
        timeline = []
        ap_time_ranges = []
        i = j = current_time = 0
        while i < len(text) and j < len(durations):
            if i < len(text) - 1 and text[i] == "A" and text[i + 1] == "P":
                start = current_time
                end = current_time + durations[j]
                timeline.append(("AP", start, end))
                ap_time_ranges.append((start, end))
                current_time = end
                j += 1
                i += 2
            else:
                char_end = current_time + durations[j]
                timeline.append(("CHAR", text[i], current_time, char_end))
                current_time = char_end
                j += 1
                i += 1

        segments = []
        last_ap_end = 0.0
        current_segment = None

        for item in timeline:
            if item[0] == "AP":
                if current_segment:
                    current_segment["end"] = item[1]
                    segments.append(current_segment)
                    current_segment = None
                last_ap_end = item[2]
            else:
                if not current_segment:
                    current_segment = {"text": "", "start": last_ap_end, "end": last_ap_end}
                current_segment["text"] += item[1]
                current_segment["end"] = item[3]

        if current_segment:
            segments.append(current_segment)
        return segments, ap_time_ranges

    def process_message(self, message):
        """处理请求的主函数"""
        name = message.content.get("name")
        dir_path = "dataset/mad_svc/lyrics_annotation"
        script_path = "dataset/mad_svc/script.txt"

        # 加载原始数据
        timestamp_json_path = os.path.join(dir_path, f"{name}.json")
        with open(timestamp_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(script_path, "r", encoding="utf-8") as f:
            script = f.read()
        data["text"] = script
        original_text = script
        durations = list(map(float, data["notes_duration"].split(" | ")))

        # 解析原始文本
        original_segments, ap_time_ranges = self.parse_text_to_segments(original_text, durations)

        merged_segments = original_segments.copy()


        translated_segments = []
        for idx, seg in enumerate(merged_segments, 1):
            if seg["end"] <= seg["start"]:
                continue
            # 保留原始时间戳
            translated_segments.append({
                "text": seg["text"],
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3)
            })
            time.sleep(0.5)

        # 生成超长 AP 分割点
        ap_chunks = []
        for start, end in ap_time_ranges:
            duration = end - start
            if duration > 12:
                current = start
                while current < end:
                    current += 12
                    if current > end:
                        current = end
                    ap_chunks.append({
                        "timestamp": current,
                        "content": "bgm"
                    })
                    if current == end:
                        break

        # 生成文本段落 chunks
        text_chunks = [{
            "timestamp": seg["end"],
            "content": seg["text"]
        } for seg in translated_segments]

        # 合并并排序所有 chunks
        all_entries = []
        for chunk in text_chunks:
            all_entries.append((chunk["timestamp"], chunk["content"], "text"))
        for chunk in ap_chunks:
            all_entries.append((chunk["timestamp"], chunk["content"], "ap"))

        # 按时间戳排序，同时确保相同时间戳时文本在前
        sorted_entries = sorted(all_entries, key=lambda x: (x[0], x[2] != "text"))

        # 生成最终格式
        chunks = [{
            "id": idx,
            "timestamp": ts,
            "content": content
        } for idx, (ts, content, _) in enumerate(sorted_entries, 1)]

        # 输出结果
        output = {
            "sentence_data": {
                "count": len(chunks),
                "chunks": chunks
            }
        }

        video_gen = 'dataset/video_edit/voice_gen'
        os.makedirs(video_gen, exist_ok=True)

        output_path = os.path.join(video_gen, 'gen_audio_timestamps.json')
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        return {"status": "success"}
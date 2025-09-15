import subprocess
import json
from pathlib import Path

from pydub import AudioSegment

from environment.agents.base import BaseAgent


class CrossTalkSubtitle(BaseAgent):
    def __init__(self):
        super().__init__()
        self.temp_srt = Path("subs.srt")
        self.segments = []

    def _format_timestamp(self, seconds: float) -> str:
        """将秒数转换为SRT标准时间格式 (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds_remainder = seconds % 60
        secs = int(seconds_remainder)
        milliseconds = int((seconds_remainder - secs) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def _get_wav_duration(self, wav_path: Path) -> float:
        """获取音频文件时长（支持多种格式）"""
        try:
            audio = AudioSegment.from_file(str(wav_path))
            duration = len(audio) / 1000.0  # 转换为秒
            return duration

        except Exception as e:
            error_msg = f"❌ 无法解析音频文件 [{wav_path.name}]: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg)
    def _generate_segments_from_json(self, json_path: Path, audio_dir: Path):
        """从JSON文件生成字幕分段，并根据对应的WAV文件计算时间"""
        with open(json_path, 'r', encoding='utf-8') as f:
            subtitles = json.load(f)
        print(subtitles)
        self.segments = []
        current_start = 0.0

        for i, item in enumerate(subtitles):
            wav_file = audio_dir / f"{i}.wav"
            print(wav_file)
            if not wav_file.exists():
                raise FileNotFoundError(f"WAV file not found: {wav_file}")
            duration = self._get_wav_duration(wav_file)
            self.segments.append({
                'start': current_start,
                'end': current_start + duration,
                'text': item['text']
            })
            current_start += duration

    def _generate_srt_from_segments(self):
        """根据分段结果生成SRT字幕文件"""
        with open(self.temp_srt, 'w', encoding='utf-8') as f:
            for idx, seg in enumerate(self.segments, 1):
                start = self._format_timestamp(seg['start'])
                end = self._format_timestamp(seg['end'])
                text = seg['text'].strip().replace('\n', ' ')
                f.write(f"{idx}\n{start} --> {end}\n{text}\n\n")

    def _burn_subtitles(self, input_video: Path, output_video: Path):
        """使用FFmpeg将字幕烧录到视频"""
        if not self.temp_srt.exists():
            raise FileNotFoundError("SRT字幕文件未生成")

        # 构建FFmpeg命令
        filter_str = (
            f"subtitles={self.temp_srt.name}:force_style="
            "'FontName=Microsoft YaHei,FontSize=12,"
            "PrimaryColour=&HFFFFFF,OutlineColour=&H000000,"
            "BackColour=&H80000000,MarginV=30'"
        )
        cmd = [
            'ffmpeg',
            '-y',
            '-i', str(input_video),
            '-vf', filter_str,
            '-c:a', 'copy',
            '-movflags', '+faststart',  # 优化MP4格式
            str(output_video)
        ]

        # 执行命令
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

    def process_message(self, message):
        """完整处理流程: 从JSON生成字幕 -> 烧录 -> 清理"""
        audio_dir = Path(message.content["audio_dir"])
        json_path = Path(message.content["json_path"])
        video_path = Path(message.content["video_path"])
        output_path = Path(message.content["output_path"])
        print(audio_dir)
        print(json_path)
        print(video_path)
        print(output_path)
        if audio_dir is None or json_path is None:
            raise ValueError("Both audio_dir and json_path must be provided")

        # 1. 从JSON生成字幕分段
        self._generate_segments_from_json(json_path, audio_dir)

        # 2. 生成SRT文件
        self._generate_srt_from_segments()

        # 3. 烧录字幕到视频
        self._burn_subtitles(video_path, output_path)

        # 4. 清理临时文件
        self.temp_srt.unlink(missing_ok=True)

        return {"status": "success", "output_path": str(output_path)}
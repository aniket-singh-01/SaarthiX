import subprocess
import whisper
from pathlib import Path
import wave
from datetime import timedelta
from openai import OpenAI

from environment.agents.base import BaseAgent
from environment.config.config import config

client = OpenAI(api_key='<KEY>')
class MadTTSSubtitleV1(BaseAgent):
    def __init__(self):
        super().__init__()
        self.temp_srt = Path("subs.srt")

    def _get_wav_duration(self, wav_path):
        with wave.open(str(wav_path), 'rb') as wav:
            return wav.getnframes() / wav.getframerate()

    def _format_timestamp(self, seconds):
        """修复时间格式：保证小时部分始终存在且为两位数"""
        td = timedelta(seconds=seconds)
        total_seconds = td.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        return f"{hours:02}:{minutes:02}:{seconds:06.3f}".replace('.', ',').zfill(12)

    def _generate_srt(self, wav_dir, txt_path):
        wav_files = sorted(
            [f for f in wav_dir.glob("*.wav") if f.stem.isdigit()],
            key=lambda x: int(x.stem)
        )

        with txt_path.open('r', encoding='utf-8') as f:
            texts = f.read().splitlines()

        if len(wav_files) != len(texts):
            raise ValueError("WAV文件数量与文本行数不匹配")

        srt_content = []
        current_time = 0.0
        for idx, (wav, text) in enumerate(zip(wav_files, texts)):
            duration = self._get_wav_duration(wav)
            end_time = current_time + duration

            # 保证时间戳格式为 00:00:00,000
            start_str = self._format_timestamp(current_time)
            end_str = self._format_timestamp(end_time)

            srt_content.append(
                f"{idx + 1}\n"
                f"{start_str} --> {end_str}\n"
                f"{text}\n"
            )
            current_time = end_time

        # 明确指定UTF-8编码，不带BOM
        with self.temp_srt.open('w', encoding='utf-8', errors='strict') as f:
            f.write('\n'.join(srt_content))

    def _burn_subtitles(self, input_video, output_video):
        if not self.temp_srt.exists():
            raise FileNotFoundError(f"字幕文件不存在: {self.temp_srt}")

        # 使用相对路径时确保文件路径正确
        cmd = [
            'ffmpeg',
            '-y',
            '-i', str(input_video),
            '-vf', f"subtitles={self.temp_srt.name}:force_style='"
                   f"FontName=Microsoft YaHei,FontSize=9,"
                   f"PrimaryColour=&HFFFFFF&,"
                   f"OutlineColour=&H000000&,"
                   f"MarginV=20,"
                   f"BorderStyle=1,Outline=0.5'",
            '-c:a', 'copy',
            str(output_video)
        ]

        try:
            # 在临时文件所在目录执行命令
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                cwd=str(self.temp_srt.parent)  # 确保工作目录正确
            )
            print("FFmpeg output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print("FFmpeg error:", e.stderr)
            raise RuntimeError(f"FFmpeg执行失败: {e.stderr}")

    def process_message(self, message):
        # 转换路径为Path对象
        derivative_dir = Path(message.content.get("derivative_dir"))
        txt_path = Path(message.content.get("txt_path"))
        video_path = Path(message.content.get("video_path"))
        output_path = Path(message.content.get("output_path"))
        print(derivative_dir)
        print(txt_path)
        print(video_path)
        print(output_path)

        # 参数验证
        if not derivative_dir.exists():
            raise FileNotFoundError(f"WAV目录不存在: {derivative_dir}")
        if not txt_path.exists():
            raise FileNotFoundError(f"文本文件不存在: {txt_path}")
        if not video_path.exists():
            raise FileNotFoundError(f"输入视频不存在: {video_path}")

        # 生成字幕文件
        self._generate_srt(derivative_dir, txt_path)

        # 烧录字幕到视频
        self._burn_subtitles(video_path, output_path)

        # 清理临时文件（可选）
        self.temp_srt.unlink(missing_ok=True)
        print(f"字幕视频已生成: {output_path}")


class MadTTSSubtitleV2(BaseAgent):
    def __init__(self):
        super().__init__()
        self.temp_srt = Path("subs.srt")
        self.segments = []
        client.api_key = config['llm']['api_key']
        client.base_url = config['llm']['base_url']

    def _format_timestamp(self, seconds: float) -> str:
        """将秒数转换为SRT标准时间格式 (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds_remainder = seconds % 60
        secs = int(seconds_remainder)
        milliseconds = int((seconds_remainder - secs) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def whisper_transcription(self, audio_path: str):
        """使用Whisper进行音频转录并保存时间戳分段"""
        model = whisper.load_model("turbo")
        result = model.transcribe(str(audio_path))
        self.segments = result.get('segments', [])


    def subtitle_correct(self, correct_path, text_list):
        with open(correct_path, 'r', encoding='utf-8') as f:
            correct_text = f.read()

        template = ""
        for i, text in enumerate(text_list, start=1):
            template += f"{i}. 待修正的文本：\n   {text}\n   修正后的文本：\n\n"

        prompt = f"""
                你是一个文本修正助手.我利用模型自动提取音频文本，但是提取的文本不够准确（可能存在个别字错误的情况），希望你能帮我修正文本。
            
                我将提供给你多行提取的文本，以及正确的完整音频文本，请你从完整文本中寻找正确的片段，按行修改提取的文本。
                
                完整的文本：
                {correct_text}
                
                输出格式：
                {template}

                具体要求:
                - 只修正错别字！
                - 如果没有错别字就不需要修改内容
                - 不要输出额外解释
                """

        response = client.chat.completions.create(
            model="claude-3-7-sonnet-20250219",
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        generated_text = response.choices[0].message.content
        extract_prompt = f"""
            你是一个文本提取助手。我将提供如下格式的文本，你需要帮我提取修正后的文本。
            
            文本格式：
            1. 待修正的文本：
               第一行文本
               修正后的文本：这是修正后的第一行
            2. 待修正的字幕：
               第二行文本
               修正后的文本：这是修正后的第二行
            。。。
            
            待提取的文本：
            {generated_text}
            
            输出格式：
            按行输出修正的文本，输出前后不要添加其余内容、解释和符号等。
            
            输出示例：
            修正后的第一行
            修正后的第二行
            。。。
        """
        response = client.chat.completions.create(
            model="deepseek-v3",
            messages=[
                {"role": "user", "content": extract_prompt}
            ],
        )
        extract_text = response.choices[0].message.content
        return extract_text

    def _generate_srt_from_whisper(self, correct_path):
        """根据Whisper的分段结果生成SRT字幕文件"""
        text_list = []
        for _, seg in enumerate(self.segments, 1):
            text = seg['text'].strip()
            text_list.append(text)

        extract_text = self.subtitle_correct(correct_path, text_list)
        corrected_lines = [line.strip() for line in extract_text.strip().split('\n') if line.strip()]

        with open(self.temp_srt, 'w', encoding='utf-8') as f:
            for idx, seg in enumerate(self.segments, 1):
                if idx > len(corrected_lines)-1:
                    break
                start = self._format_timestamp(seg['start'])
                end = self._format_timestamp(seg['end'])
                text = corrected_lines[idx - 1].strip().replace('\n', ' ')
                f.write(f"{idx}\n{start} --> {end}\n{text}\n\n")

    def _burn_subtitles(self, input_video: Path, output_video: Path):
        """使用FFmpeg将字幕烧录到视频"""
        if not self.temp_srt.exists():
            raise FileNotFoundError("SRT字幕文件未生成")

        # 构建FFmpeg命令
        filter_str = (
            f"subtitles={self.temp_srt.name}:force_style="
            "'FontName=Microsoft YaHei,FontSize=10,"
            "PrimaryColour=&HFFFFFF,OutlineColour=&H000000,"
            "BackColour=&H80000000,MarginV=20'"
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
        """完整处理流程: 转录 -> 生成字幕 -> 烧录 -> 清理"""
        txt_path = Path(message.content.get("txt_path"))
        video_path = Path(message.content["video_path"])
        output_path = Path(message.content["output_path"])

        # 1. 使用Whisper生成字幕
        self.whisper_transcription(str(video_path))  # 直接处理视频文件

        # 2. 生成SRT文件
        self._generate_srt_from_whisper(txt_path)

        # 3. 烧录字幕到视频
        self._burn_subtitles(video_path, output_path)

        # 4. 清理临时文件
        self.temp_srt.unlink(missing_ok=True)

        return {"status": "success", "output_path": str(output_path)}
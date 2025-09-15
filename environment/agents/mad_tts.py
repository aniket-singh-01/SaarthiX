import os
import subprocess

import yaml

from environment.communication.message import Message
from environment.roles.loudness_normalizer import LoudnessNormalizer
from environment.roles.mad_tts.mad_tts_combiner import MadTTSCombiner
from environment.roles.mad_tts.mad_tts_infer import MadTTSInfer
from environment.roles.mad_tts.mad_tts_slicer import MadTTSSlicer
from environment.roles.mad_tts.mad_tts_subtitle import MadTTSSubtitleV1,MadTTSSubtitleV2
from environment.roles.mad_tts.mad_tts_writer import MadTTSWriter
from environment.roles.resampler import Resampler
from environment.roles.separator import Separator
from environment.roles.transcriber import Transcriber



class MadTTSAgent:
    def __init__(self, config):
        self.video_path = config["mad_tts"]["video_path"]
        self.reqs = config["mad_tts"]["reqs"]
        self.audio_path = self.extract_audio()

        self.separator = Separator()
        self.normalizer = LoudnessNormalizer()
        self.resampler = Resampler()
        self.slicer = MadTTSSlicer()
        self.transcriber = Transcriber()
        self.writer = MadTTSWriter()
        self.infer = MadTTSInfer()
        self.combiner = MadTTSCombiner()
        self.subtitle_v1 = MadTTSSubtitleV1()
        self.subtitle_v2 = MadTTSSubtitleV2()

    def extract_audio(self):
        audio_path = os.path.splitext(self.video_path)[0] + ".wav"

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # 覆盖已存在文件
            "-i", self.video_path,
            "-vn",  # 禁用视频处理
            "-acodec", "pcm_s16le",  # 16-bit PCM编码
            "-ar", "44100",  # 采样率
            "-ac", "2",  # 立体声
            "-loglevel", "error",  # 仅显示错误信息
            audio_path
        ]

        try:
            subprocess.run(ffmpeg_cmd, check=True)
            print(f"Audio extracted to: {audio_path}")
            return audio_path

        except subprocess.CalledProcessError as e:
            print(f"Conversion failed: {e.stderr.decode()}")
        except FileNotFoundError:
            print("Error: FFmpeg not found. Please install FFmpeg and add it to the system PATH")

    def orchestrator(self):
        pre_msg = Message(content={"audio_dir": os.path.dirname(self.audio_path)})

        separator_result = self.separator.process_message(pre_msg)

        normalizer_result = self.normalizer.process_message(pre_msg)

        resampler_result = self.resampler.process_message(pre_msg)

        transcriber_result = self.transcriber.process_message(pre_msg)

        slicer_msg = Message(content={"audio_path": self.audio_path, "output_dir": os.path.splitext(self.audio_path)[0]})
        slicer_result = self.slicer.process_message(slicer_msg)

        msg = Message(content={"audio_dir": os.path.splitext(self.audio_path)[0]})
        transcriber_result = self.transcriber.process_message(msg)

        lab_path = os.path.splitext(self.audio_path)[0] + ".lab"
        writer_msg = Message(content={"lab_path": lab_path, "reqs": self.reqs})
        writer_result = self.writer.process_message(writer_msg)

        infer_result = self.infer.process_message(msg)

        combiner_msg = Message(content={"video_path": self.video_path, "audio_dir": os.path.splitext(self.audio_path)[0]})
        combiner_result = self.combiner.process_message(combiner_msg)

        # derivative_dir = os.path.join(os.path.splitext(self.audio_path)[0], "derivative")
        # txt_path = os.path.join(os.path.dirname(self.audio_path), "speech.txt")
        # video_path = os.path.join(os.path.splitext(self.audio_path)[0], "final", "final.mp4")
        # output_path = os.path.join(os.path.dirname(video_path), "final_subtitle.mp4")
        # subtitle_msg = Message(content={"derivative_dir": derivative_dir, "txt_path": txt_path, "video_path": video_path, "output_path": output_path})
        # subtitle_result = self.subtitle_v2.process_message(subtitle_msg)
        return 1

def gen_mad_tts():
    print("Welcome to the Mad Generator TTS")
    with open('environment/config/mad_tts.yml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(config)
    agent = MadTTSAgent(config)
    return agent.orchestrator()

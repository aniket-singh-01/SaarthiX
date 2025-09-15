

import json
import os
import subprocess
import sys

from environment.agents.base import BaseAgent
from environment.communication.message import Message


class MadTTSCombiner(BaseAgent):
    def __init__(self):
        super().__init__()
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

    def process_message(self, message):
        try:
            video_path = message.content.get("video_path")
            audio_dir = message.content.get("audio_dir")

            if not video_path or not os.path.isfile(video_path):
                raise FileNotFoundError(f"视频文件不存在: {video_path}")
            if not audio_dir or not os.path.isdir(audio_dir):
                raise FileNotFoundError(f"音频目录不存在: {audio_dir}")

            metadata_path = os.path.join(audio_dir, 'metadata.json')
            derivative_dir = os.path.join(audio_dir, 'derivative')
            output_dir = os.path.join(audio_dir, 'final')
            final_output = os.path.join(output_dir, 'final.mp4')

            os.makedirs(output_dir, exist_ok=True)

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            processed_files = []

            for clip in metadata:
                file_name = clip['file']
                derivative_audio = os.path.join(derivative_dir, file_name)

                if not os.path.exists(derivative_audio):
                    continue

                clip_path = os.path.join(output_dir, f"clip_{file_name.replace('.wav', '.mp4')}")
                cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(clip['start']),
                    '-i', video_path,
                    '-to', str(clip['end'] - clip['start']),
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-vf', 'scale=iw:ih',
                    clip_path
                ]
                subprocess.run(cmd, check=True)

                cmd = [
                    'ffprobe',
                    '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    derivative_audio
                ]
                target_duration = float(subprocess.run(cmd, stdout=subprocess.PIPE, text=True).stdout.strip())

                adjusted_path = os.path.join(output_dir, f"adjusted_{file_name.replace('.wav', '.mp4')}")
                speed_factor = target_duration / clip['duration']
                cmd = [
                    'ffmpeg', '-y',
                    '-i', clip_path,
                    '-filter:v', f'setpts={speed_factor}*PTS',
                    '-an',
                    adjusted_path
                ]
                subprocess.run(cmd)

                merged_path = os.path.join(output_dir, f"merged_{file_name.replace('.wav', '.mp4')}")
                cmd = [
                    'ffmpeg', '-y',
                    '-i', adjusted_path,
                    '-i', derivative_audio,
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-map', '0:v:0',
                    '-map', '1:a:0',
                    '-shortest',
                    merged_path
                ]
                subprocess.run(cmd)
                processed_files.append(merged_path)

            with open(os.path.join(output_dir, 'filelist.txt'), 'w', encoding='utf-8') as f:
                for file in processed_files:
                    f.write(f"file '{os.path.abspath(file)}'\n")

            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', os.path.join(output_dir, 'filelist.txt'),
                '-c', 'copy',
                final_output
            ]
            subprocess.run(cmd)

            print("正在清理中间文件...")
            for filename in os.listdir(output_dir):
                file_path = os.path.join(output_dir, filename)
                if filename != os.path.basename(final_output) and os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                        print(f"已删除: {filename}")
                    except Exception as e:
                        print(f"删除失败 [{filename}]: {str(e)}")

            print(f"处理完成！最终视频保存在: {final_output}")

            return Message(content={"status": "success"})

        except Exception as e:
            return Message(content={"status": "error", "message": f"处理失败: {str(e)}"})


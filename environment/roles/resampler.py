import os
import subprocess
import sys
import threading
from pathlib import Path

from ..agents.base import BaseAgent
from ..communication.message import Message


class Resampler(BaseAgent):
    def __init__(self):
        super().__init__()

    def _read_output(self, pipe):
        """读取子进程输出的辅助函数"""
        for line in iter(pipe.readline, ''):
            if line:
                print(line.strip(), flush=True)
        pipe.close()

    def process_message(self, message):
        audio_dir = message.content.get("audio_dir")
        print(f"原始音频路径: {audio_dir}")

        if not os.path.isabs(audio_dir):
            abs_audio_dir = os.path.abspath(audio_dir)
            print(f"转换为绝对路径: {abs_audio_dir}")
        else:
            abs_audio_dir = audio_dir

        print(f"音频路径存在: {os.path.exists(abs_audio_dir)}")

        cmd = ["fap", "resample", str(abs_audio_dir), str(abs_audio_dir), "--overwrite"]
        cmd_str = " ".join(cmd)
        print(f"执行命令: {cmd_str}")

        try:
            process = subprocess.Popen(
                cmd_str,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='gbk',
                bufsize=1
            )

            print("开始重采样，实时输出：")

            stdout_thread = threading.Thread(target=self._read_output, args=(process.stdout,))
            stderr_thread = threading.Thread(target=self._read_output, args=(process.stderr,))

            stdout_thread.daemon = True
            stderr_thread.daemon = True

            stdout_thread.start()
            stderr_thread.start()

            return_code = process.wait()

            stdout_thread.join()
            stderr_thread.join()

            print(f"重采样完成，返回码: {return_code}")

            return Message(
                content={
                    "status": "success" if return_code == 0 else "error",
                    "message": "Audio resample completed successfully" if return_code == 0
                    else f"Audio resample failed with return code {return_code}"
                }
            )

        except Exception as e:
            print(f"重采样出错: {str(e)}")
            return Message(
                content={
                    "status": "error",
                    "message": f"Audio resample failed: {str(e)}"
                }
            )

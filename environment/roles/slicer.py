import os
import subprocess
import sys
import threading
from pathlib import Path

from ..agents.base import BaseAgent
from ..communication.message import Message


class Slicer(BaseAgent):
    def __init__(self):
        super().__init__()
        scripts_dir = os.path.join(sys.prefix, 'Scripts')
        fap_path = os.path.join(scripts_dir, 'fap.exe')
        self.fap_path = fap_path
        # 验证路径是否存在
        if not os.path.exists(self.fap_path):
            print(f"警告: fap.exe 不存在于路径 {self.fap_path}")

    def _read_output(self, pipe):
        """读取子进程输出的辅助函数"""
        for line in iter(pipe.readline, ''):
            if line:
                print(line.strip(), flush=True)
        pipe.close()

    def process_message(self, message):
        # 获取音频路径
        audio_dir = message.content.get("audio_dir")
        print(f"原始音频路径: {audio_dir}")

        # 处理音频路径，确保使用绝对路径
        if not os.path.isabs(audio_dir):
            abs_audio_dir = os.path.abspath(audio_dir)
            print(f"转换为绝对路径: {abs_audio_dir}")
        else:
            abs_audio_dir = audio_dir

        # 检查路径是否存在
        print(f"音频路径存在: {os.path.exists(abs_audio_dir)}")

        # 构建命令，使用完整路径
        cmd = [self.fap_path, "slice-audio-v2", str(abs_audio_dir), str(abs_audio_dir), "--overwrite"]
        cmd_str = " ".join(cmd)
        print(f"执行命令: {cmd_str}")

        try:
            # 使用Popen开始进程
            process = subprocess.Popen(
                cmd_str,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',  # 如果遇到编码问题，可以改为'gbk'
                bufsize=1
            )

            print("开始切片，实时输出：")

            # 创建线程读取输出
            stdout_thread = threading.Thread(target=self._read_output, args=(process.stdout,))
            stderr_thread = threading.Thread(target=self._read_output, args=(process.stderr,))

            # 设置为守护线程，这样主程序退出时它们也会退出
            stdout_thread.daemon = True
            stderr_thread.daemon = True

            # 启动线程
            stdout_thread.start()
            stderr_thread.start()

            # 等待进程完成
            return_code = process.wait()

            # 等待线程读取完所有输出
            stdout_thread.join()
            stderr_thread.join()

            print(f"切片完成，返回码: {return_code}")

            # 返回处理结果
            return Message(
                content={
                    "status": "success" if return_code == 0 else "error",
                    "message": "Audio Slice completed successfully" if return_code == 0
                    else f"Audio Slice failed with return code {return_code}"
                }
            )

        except Exception as e:
            print(f"切片出错: {str(e)}")
            return Message(
                content={
                    "status": "error",
                    "message": f"Audio Slice failed: {str(e)}"
                }
            )
import os
import sys
import subprocess
from environment.agents.base import BaseAgent
from environment.communication.message import Message


class MadSVCCoverist(BaseAgent):
    def __init__(self):
        super().__init__()

    def process_message(self, message):
        source = '../../' + message.content.get('source')
        target = '../../' + message.content.get('target')
        output = '../../dataset/video_edit/voice_gen'
        print(f"Source: {source}")
        print(f"Target: {target}")

        original_dir = os.getcwd()
        original_pythonpath = os.environ.get("PYTHONPATH", "")

        try:
            # 切换到 seedvc 目录
            seedvc_dir = os.path.join("tools", "seed-vc")
            os.chdir(seedvc_dir)
            os.makedirs(output, exist_ok=True)

            seedvc_abs_path = os.path.abspath('.')
            os.environ["PYTHONPATH"] = seedvc_abs_path


            cmd_parts = [
                sys.executable,
                "inference.py",
                "--source", source,
                "--target", target,
                "--output", output,
                "--f0-condition", "True"
            ]

            try:
                process = subprocess.run(cmd_parts, capture_output=True, text=True, encoding='utf-8')
            except UnicodeDecodeError:
                process = subprocess.run(cmd_parts, capture_output=True, text=False)
                stdout = process.stdout.decode('utf-8', errors='replace') if process.stdout else ""
                stderr = process.stderr.decode('utf-8', errors='replace') if process.stderr else ""
                process.stdout = stdout
                process.stderr = stderr

            print(f"标准输出: {process.stdout}")
            if process.stderr:
                print(f"错误输出: {process.stderr}")

            if process.returncode == 0:
                print("命令执行成功")
                return Message(content={'status': 'success', 'output_dir': 'dataset/video_edit/voice_gen'})
            else:
                print(f"命令执行失败，返回码: {process.returncode}")
                return Message(content={
                    'status': 'error',
                    'error': f"命令执行失败，返回码: {process.returncode}",
                    'stderr': process.stderr
                })

        except Exception as e:
            print(f"执行过程中发生错误: {e}")
            return Message(content={'status': 'error', 'error': str(e)})

        finally:
            os.chdir(original_dir)

            if original_pythonpath:
                os.environ["PYTHONPATH"] = original_pythonpath
            else:
                if "PYTHONPATH" in os.environ:
                    del os.environ["PYTHONPATH"]

            print(f"已恢复工作目录: {original_dir}")
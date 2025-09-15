import argparse
import os
import sys


def run_diffsinger(exp_name="0228_opencpop_ds100_rel", input_dir=""):
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run diff singer inference')
    parser.add_argument('--exp_name', default=exp_name, type=str, help='Experiment name')
    parser.add_argument('--input_dir', default=input_dir, type=str, help='audio info')
    args = parser.parse_args([])  # Empty list to avoid reading command line arguments

    # Get current working directory
    original_dir = os.getcwd()
    original_sys_path = sys.path.copy()  # 保存原始sys.path

    try:
        # Change to DiffSinger directory
        diffsinger_dir = os.path.join("tools", "DiffSinger")
        os.chdir(diffsinger_dir)

        # 添加DiffSinger目录到Python路径
        diffsinger_abs_path = os.path.abspath('.')
        sys.path.append(diffsinger_abs_path)

        # 或者直接修改PYTHONPATH环境变量
        os.environ["PYTHONPATH"] = diffsinger_abs_path

        # 使用subprocess运行
        import subprocess
        cmd_parts = [
            sys.executable,
            "inference/svs/ds_e2e.py",
            "--exp_name", args.exp_name,
            "--input_dir", args.input_dir,
        ]

        process = subprocess.run(cmd_parts)
        return_code = process.returncode
        print(f"Process returned: {return_code}")
        return return_code

    finally:
        # 恢复原始环境
        os.chdir(original_dir)
        sys.path = original_sys_path  # 恢复原始sys.path
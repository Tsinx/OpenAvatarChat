#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动脚本，设置UTF-8编码并运行OpenAvatarChat V2 - 手动录音控制版本
"""

import os
import sys
import subprocess

def main():
    # 设置UTF-8编码环境变量
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    env['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'
    
    # 获取命令行参数
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    # 构建命令 - 使用uv run
    cmd = ['uv', 'run', 'demo_v2.py'] + args
    
    print(f"Running OpenAvatarChat V2 - Manual Recording Control")
    print(f"Command: {' '.join(cmd)}")
    print(f"Environment variables set:")
    print(f"  PYTHONIOENCODING=utf-8")
    print(f"  PYTHONUTF8=1")
    print(f"  PYTHONLEGACYWINDOWSSTDIO=utf-8")
    print(f"\nFeatures:")
    print(f"  - Manual recording control (hold to record, release to send)")
    print(f"  - No VAD auto-detection")
    print(f"  - More accurate speech input")
    print()
    
    # 运行程序
    try:
        result = subprocess.run(cmd, env=env, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running program: {e}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        return 0

if __name__ == "__main__":
    exit(main())

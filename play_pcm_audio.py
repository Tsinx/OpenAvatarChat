#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCM音频文件播放脚本 - 增强版
用于播放火山引擎TTS生成的PCM音频文件
"""

import pyaudio
import wave
import sys
import os
from pathlib import Path

def play_pcm_file_with_params(pcm_file_path, sample_rate=16000, channels=1, sample_width=2, format_type='int16'):
    """
    使用指定参数播放PCM音频文件
    
    Args:
        pcm_file_path (str): PCM文件路径
        sample_rate (int): 采样率
        channels (int): 声道数
        sample_width (int): 采样位宽（字节）
        format_type (str): 数据格式类型
    """
    if not os.path.exists(pcm_file_path):
        print(f"错误：文件不存在 - {pcm_file_path}")
        return False
    
    try:
        # 初始化PyAudio
        p = pyaudio.PyAudio()
        
        # 根据format_type确定PyAudio格式
        if format_type == 'int16':
            pa_format = pyaudio.paInt16
        elif format_type == 'int24':
            pa_format = pyaudio.paInt24
        elif format_type == 'int32':
            pa_format = pyaudio.paInt32
        elif format_type == 'float32':
            pa_format = pyaudio.paFloat32
        else:
            pa_format = p.get_format_from_width(sample_width)
        
        # 打开音频流
        stream = p.open(
            format=pa_format,
            channels=channels,
            rate=sample_rate,
            output=True
        )
        
        print(f"播放参数: 采样率={sample_rate}Hz, 声道数={channels}, 位宽={sample_width*8}位, 格式={format_type}")
        
        # 读取并播放PCM数据
        chunk_size = 1024
        with open(pcm_file_path, 'rb') as f:
            data = f.read(chunk_size)
            while data:
                stream.write(data)
                data = f.read(chunk_size)
        
        print("播放完成")
        
        # 清理资源
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        return True
        
    except Exception as e:
        print(f"播放PCM文件时出错: {e}")
        return False

def try_different_parameters(pcm_file_path):
    """
    尝试不同的音频参数来播放PCM文件
    """
    # 常见的音频参数组合 - 基于火山引擎官方文档
    param_combinations = [
        # 火山引擎TTS输出格式 - 官方推荐
        {'sample_rate': 24000, 'channels': 1, 'sample_width': 4, 'format_type': 'float32'},  # 32bit位深
        {'sample_rate': 24000, 'channels': 1, 'sample_width': 2, 'format_type': 'int16'},  # 16bit位深 (pcm_s16le)
        
        # 火山引擎ASR输入格式
        {'sample_rate': 16000, 'channels': 1, 'sample_width': 2, 'format_type': 'int16'},
        
        # 其他常见参数（用于兼容性测试）
        {'sample_rate': 8000, 'channels': 1, 'sample_width': 2, 'format_type': 'int16'},
        {'sample_rate': 32000, 'channels': 1, 'sample_width': 2, 'format_type': 'int16'},
        {'sample_rate': 44100, 'channels': 1, 'sample_width': 2, 'format_type': 'int16'},
        {'sample_rate': 48000, 'channels': 1, 'sample_width': 2, 'format_type': 'int16'},
        {'sample_rate': 16000, 'channels': 2, 'sample_width': 2, 'format_type': 'int16'},
    ]
    
    print(f"\n尝试不同参数播放文件: {pcm_file_path}")
    print("请听每个版本的播放效果，选择音质最好的参数\n")
    
    for i, params in enumerate(param_combinations, 1):
        print(f"\n=== 尝试参数组合 {i} ===")
        input("按回车键开始播放...")
        
        success = play_pcm_file_with_params(pcm_file_path, **params)
        if success:
            choice = input("这个参数组合的音质如何？(好/差/退出): ").strip().lower()
            if choice in ['好', 'good', 'y', 'yes']:
                print(f"\n找到合适的参数组合:")
                for key, value in params.items():
                    print(f"  {key}: {value}")
                return params
            elif choice in ['退出', 'quit', 'exit', 'q']:
                break
    
    return None

def analyze_pcm_file(pcm_file_path):
    """
    分析PCM文件的基本信息
    """
    if not os.path.exists(pcm_file_path):
        print(f"文件不存在: {pcm_file_path}")
        return
    
    file_size = os.path.getsize(pcm_file_path)
    print(f"\n=== PCM文件分析 ===")
    print(f"文件路径: {pcm_file_path}")
    print(f"文件大小: {file_size} 字节")
    
    # 基于文件大小估算可能的参数
    print(f"\n可能的播放时长估算:")
    for rate in [8000, 16000, 22050, 24000, 44100, 48000]:
        for channels in [1, 2]:
            for width in [1, 2, 3, 4]:
                duration = file_size / (rate * channels * width)
                if 0.1 <= duration <= 300:  # 合理的时长范围
                    print(f"  {rate}Hz, {channels}声道, {width*8}位: {duration:.2f}秒")

def main():
    """主函数"""
    # 默认的PCM文件路径
    default_pcm_file = r"C:\Users\Administrator\AppData\Local\Temp\volcengine_tts_debug_20250903_162603_202435.pcm"
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        pcm_file_path = sys.argv[1]
    else:
        pcm_file_path = default_pcm_file
    
    print(f"PCM音频播放工具")
    print(f"目标文件: {pcm_file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(pcm_file_path):
        print(f"\n错误：文件不存在 - {pcm_file_path}")
        print("\n使用方法:")
        print(f"  python {sys.argv[0]} [PCM文件路径]")
        return
    
    # 分析文件
    analyze_pcm_file(pcm_file_path)
    
    print("\n选择操作:")
    print("1. 使用默认参数播放 (16kHz, 单声道, 16位)")
    print("2. 尝试不同参数组合找到最佳音质")
    print("3. 手动指定参数播放")
    print("4. 转换为WAV文件")
    
    try:
        choice = input("\n请选择 (1/2/3/4): ").strip()
        
        if choice == '1':
            # 默认参数播放
            play_pcm_file_with_params(pcm_file_path)
            
        elif choice == '2':
            # 尝试不同参数
            best_params = try_different_parameters(pcm_file_path)
            if best_params:
                print("\n建议保存这些参数用于以后播放相同来源的PCM文件")
                
        elif choice == '3':
            # 手动指定参数
            print("\n请输入音频参数:")
            sample_rate = int(input("采样率 (如16000): ") or "16000")
            channels = int(input("声道数 (1或2): ") or "1")
            sample_width = int(input("采样位宽字节数 (1/2/3/4): ") or "2")
            format_type = input("数据格式 (int16/int24/int32/float32): ") or "int16"
            
            play_pcm_file_with_params(pcm_file_path, sample_rate, channels, sample_width, format_type)
            
        elif choice == '4':
            # 转换为WAV
            print("\n请输入转换参数:")
            sample_rate = int(input("采样率 (如16000): ") or "16000")
            channels = int(input("声道数 (1或2): ") or "1")
            sample_width = int(input("采样位宽字节数 (1/2/3/4): ") or "2")
            
            wav_file_path = pcm_file_path.replace('.pcm', '.wav')
            
            try:
                with open(pcm_file_path, 'rb') as pcm_file:
                    pcm_data = pcm_file.read()
                
                with wave.open(wav_file_path, 'wb') as wav_file:
                    wav_file.setnchannels(channels)
                    wav_file.setsampwidth(sample_width)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(pcm_data)
                
                print(f"\n转换完成: {wav_file_path}")
                
                # 使用系统播放器播放
                import subprocess
                if sys.platform.startswith('win'):
                    os.startfile(wav_file_path)
                    
            except Exception as e:
                print(f"转换失败: {e}")
                
        else:
            print("无效选择")
            
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"操作过程中出错: {e}")

if __name__ == "__main__":
    main()
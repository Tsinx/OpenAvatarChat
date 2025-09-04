#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 VolcEngineRealtimeHandler 的 _send_temp_file_and_wait_response 方法
"""

import asyncio
import sys
import os
from pathlib import Path
from loguru import logger

# 添加项目路径到 sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from handlers.realtime.volcengine.realtime_handler_volcengine import HandlerVolcEngineRealtime, VolcEngineRealtimeContext, VolcEngineRealtimeConfig

async def test_send_temp_file():
    """测试发送临时文件方法"""
    
    # 音频文件路径
    audio_file_path = "d:\\Projects\\TA\\OpenAvatarChat\\src\\handlers\\realtime\\volcengine\\volcengine_real_time_official\\python3.7\\whoareyou.wav"
    
    # 检查文件是否存在
    if not os.path.exists(audio_file_path):
        print(f"错误: 音频文件不存在: {audio_file_path}")
        return
    
    print(f"开始测试发送音频文件: {audio_file_path}")
    
    try:
        # 创建配置
        config = VolcEngineRealtimeConfig(
            ws_url="wss://openspeech.bytedance.com/api/v3/realtime/dialogue",
            app_id="5385188715",
            access_key="7Mb2pErVhX_z2bpEjWlYJa6N9yW2RhmS",
            resource_id="volc.speech.dialog",
            app_key="PlgvMymc7f3tQnJ6",
            tts_voice="zh_male_yunzhou_jupiter_bigtts",
            sample_rate=24000,
            channels=1,
            audio_format="pcm",
            bot_name="小助手",
            system_role="你是一个友善的AI助手"
        )
        
        # 创建处理器实例
        handler = HandlerVolcEngineRealtime()
        
        # 创建上下文
        context = VolcEngineRealtimeContext(
            session_id="test_session_001",
        )
        context.config = config
        
        print("开始调用 _send_temp_file_and_wait_response 方法...")
        
        # 调用测试方法
        await handler._send_temp_file_and_wait_response(context, audio_file_path)
        
        print("测试完成!")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        print(f"错误堆栈: {traceback.format_exc()}")

if __name__ == "__main__":
    # 确保logs目录存在
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 配置日志输出到文件
    logger.remove()  # 移除默认的控制台输出
    logger.add("logs/test_send_temp_file.log", 
               rotation="10 MB", 
               retention="7 days", 
               level="DEBUG",
               format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}")
    
    # 同时保留控制台输出以便实时查看
    logger.add(sys.stderr, level="INFO")
    
    print("=== 测试 _send_temp_file_and_wait_response 方法 ===")
    print("日志将保存到: logs/test_send_temp_file.log")
    asyncio.run(test_send_temp_file())
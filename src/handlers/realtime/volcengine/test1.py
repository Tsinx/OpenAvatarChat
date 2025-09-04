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

# 添加项目路径到 sys.path - 使用绝对路径
sys.path.insert(0, r"D:\Projects\TA\OpenAvatarChat\src")
sys.path.insert(0, r"D:\Projects\TA\OpenAvatarChat\src\handlers\realtime\volcengine")

from realtime_handler_volcengine import HandlerVolcEngineRealtime, VolcEngineRealtimeContext, VolcEngineRealtimeConfig
import asyncio
import logging
import os
import queue
import tempfile
import threading
import time
import uuid
import wave
from abc import ABC
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import pyaudio
from loguru import logger
from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDetail, HandlerDataInfo
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.data_models.runtime_data.data_bundle import DataBundleDefinition, DataBundleEntry, DataBundleEntry
from pydantic import BaseModel, Field

from volcengine_client import VolcEngineRealtimeClient
from audio_processor import AudioConfig, AudioDeviceManager, DialogSession, AudioProcessor
from audio_recorder import AudioRecorder, AudioPlayer

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
        
        # 创建一个ChatData对象，模拟录音停止信号
        from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
        
        # 创建数据定义
        definition = DataBundleDefinition()
        text_entry = DataBundleEntry.create_text_entry("recording_signal")
        definition.add_entry(text_entry)
        definition.set_main_entry("recording_signal")
        
        # 创建数据包
        data_bundle = DataBundle(definition)
        data_bundle.set_data("recording_signal", "stop")
        data_bundle.add_meta("recording_signal", "stop")
        
        chat_data = ChatData(
            type=ChatDataType.HUMAN_VOICE_ACTIVITY,
            data=data_bundle
        )
        
        # 调用测试方法
        await handler.handle(context, chat_data, {})
        
        print("测试完成!")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        print(f"错误堆栈: {traceback.format_exc()}")

if __name__ == "__main__":
    # 确保logs目录存在 - 使用项目根目录
    project_root = Path(r"D:\Projects\TA\OpenAvatarChat")
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # 配置日志输出到文件
    logger.remove()  # 移除默认的控制台输出
    log_file_path = log_dir / "test_send_temp_file.log"
    logger.add(str(log_file_path), 
               rotation="10 MB", 
               retention="7 days", 
               level="DEBUG",
               format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}")
    
    # 同时保留控制台输出以便实时查看
    logger.add(sys.stderr, level="INFO")
    
    print("=== 测试 _send_temp_file_and_wait_response 方法 ===")
    print(f"日志将保存到: {log_file_path}")
    asyncio.run(test_send_temp_file())
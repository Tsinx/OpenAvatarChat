#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 VolcEngineRealtimeHandler 的音频输出质量
专门测试PCM音频是否存在噪音问题
"""

import asyncio
import sys
import os
import time
import tempfile
from pathlib import Path
from loguru import logger

# 添加项目路径到 sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from handlers.realtime.volcengine.realtime_handler_volcengine import HandlerVolcEngineRealtime, VolcEngineRealtimeContext, VolcEngineRealtimeConfig
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
from chat_engine.common.handler_base import HandlerDataInfo
from chat_engine.contexts.handler_context import HandlerContext
import numpy as np

async def test_volcengine_audio_output():
    """测试火山引擎音频输出质量"""
    
    # 音频文件路径（使用代码中的测试文件）
    audio_file_path = "src/handlers/realtime/volcengine/volcengine_real_time_official/python3.7/whoareyou.wav"
    
    # 检查文件是否存在
    if not os.path.exists(audio_file_path):
        print(f"错误: 音频文件不存在: {audio_file_path}")
        return
    
    print(f"开始测试火山引擎音频输出: {audio_file_path}")
    
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
        volcengine_context = VolcEngineRealtimeContext(
            session_id="test_audio_session_001",
        )
        volcengine_context.config = config
        
        # 创建HandlerContext
        handler_context = HandlerContext(session_id="test_audio_session_001")
        handler_context.volcengine_context = volcengine_context
        
        # 创建输入数据（模拟人类音频输入）
        # 读取测试音频文件
        with open(audio_file_path, 'rb') as f:
            audio_bytes = f.read()
        
        # 转换为numpy数组（假设是16位PCM）
        # 跳过WAV文件头（通常44字节）
        audio_data = audio_bytes[44:]  # 跳过WAV头
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_array = audio_array.astype(np.float32) / 32768.0  # 归一化
        audio_array = audio_array[np.newaxis, ...]  # 添加声道维度
        
        # 创建输入数据
        input_definition = DataBundleDefinition()
        input_definition.add_entry(DataBundleEntry.create_audio_entry("human_audio", 1, 24000))
        input_data_bundle = DataBundle(definition=input_definition)
        input_data_bundle.set_main_data(audio_array)
        
        input_chat_data = ChatData(
            source="test_input",
            type=ChatDataType.HUMAN_AUDIO,
            timestamp=time.time(),
            data=input_data_bundle
        )
        
        # 定义输出数据类型
        output_definition = DataBundleDefinition()
        output_definition.add_entry(DataBundleEntry.create_audio_entry("avatar_audio", 1, 24000))
        output_definitions = {
            ChatDataType.AVATAR_AUDIO: HandlerDataInfo(
                type=ChatDataType.AVATAR_AUDIO,
                definition=output_definition
            )
        }
        
        print("开始调用 handle 方法进行音频处理...")
        
        # 设置输出数据收集
        output_audio_data = []
        
        def collect_output_data(chat_data: ChatData):
            """收集输出数据"""
            if chat_data.type == ChatDataType.AVATAR_AUDIO:
                output_audio_data.append(chat_data)
                print(f"收到音频输出: {chat_data.data.get_main_data().shape}")
        
        # 设置数据收集回调
        volcengine_context.submit_data = collect_output_data
        
        # 调用handle方法
        await handler.handle(handler_context, input_chat_data, output_definitions)
        
        # 等待一段时间让异步处理完成
        print("等待音频处理完成...")
        await asyncio.sleep(10)  # 等待10秒
        
        # 检查输出结果
        if output_audio_data:
            print(f"\n=== 音频输出测试结果 ===")
            print(f"收到 {len(output_audio_data)} 个音频输出")
            
            for i, audio_data in enumerate(output_audio_data):
                audio_array = audio_data.data.get_main_data()
                print(f"\n音频片段 {i+1}:")
                print(f"  形状: {audio_array.shape}")
                print(f"  数据类型: {audio_array.dtype}")
                print(f"  数值范围: [{audio_array.min():.6f}, {audio_array.max():.6f}]")
                print(f"  均值: {audio_array.mean():.6f}")
                print(f"  标准差: {audio_array.std():.6f}")
                
                # 保存为PCM文件用于播放测试
                timestamp = int(time.time() * 1000)
                pcm_file = f"volcengine_output_{i+1}_{timestamp}.pcm"
                
                # 转换回16位PCM用于播放
                audio_16bit = (audio_array * 32767).astype(np.int16)
                with open(pcm_file, 'wb') as f:
                    f.write(audio_16bit.tobytes())
                
                print(f"  已保存PCM文件: {pcm_file}")
                print(f"  播放命令: ffplay -f s16le -ar 24000 -ac 1 {pcm_file}")
                
                # 简单的噪音检测
                # 检查是否有异常的高频噪音或静音
                if audio_array.std() < 0.001:
                    print(f"  ⚠️  警告: 音频可能过于安静或为静音")
                elif audio_array.std() > 0.5:
                    print(f"  ⚠️  警告: 音频可能包含噪音或过于响亮")
                else:
                    print(f"  ✅ 音频动态范围正常")
                
                # 检查削波
                clipped_samples = np.sum(np.abs(audio_array) >= 0.99)
                if clipped_samples > 0:
                    print(f"  ⚠️  警告: 检测到 {clipped_samples} 个削波样本")
                else:
                    print(f"  ✅ 无削波现象")
            
            print(f"\n=== 测试完成 ===")
            print(f"请使用 ffplay 命令播放生成的PCM文件来验证音频质量")
            
        else:
            print("❌ 未收到任何音频输出")
        
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
    logger.add("logs/test_volcengine_audio_output.log", 
               rotation="10 MB", 
               retention="7 days", 
               level="DEBUG",
               format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}")
    
    # 同时保留控制台输出以便实时查看
    logger.add(sys.stderr, level="INFO")
    
    print("=== 测试火山引擎音频输出质量 ===")
    print("日志将保存到: logs/test_volcengine_audio_output.log")
    asyncio.run(test_volcengine_audio_output())
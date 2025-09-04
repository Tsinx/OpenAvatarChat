# -*- coding: utf-8 -*-
"""
火山引擎实时对话处理器

该模块实现了火山引擎实时对话的主处理器，负责：
1. 处理音频输入和输出
2. 管理WebSocket连接和会话
3. 协调ASR、LLM和TTS的实时交互
4. 处理音频格式转换和数据流
"""

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


class VolcEngineRealtimeConfig(HandlerBaseConfigModel):
    """火山引擎实时对话配置模型"""
    
    # WebSocket连接配置
    ws_url: str = Field(default="wss://openspeech.bytedance.com/api/v3/realtime/dialogue")
    app_id: str = Field(default="")
    access_key: str = Field(default="")
    app_key: str = Field(default="")
    resource_id: str = Field(default="volc.speech.dialog")
    
    # 音频配置
    audio_format: str = Field(default="pcm")
    sample_rate: int = Field(default=16000)
    channels: int = Field(default=1)
    bits_per_sample: int = Field(default=16)
    chunk_size: int = Field(default=1024)
    
    # ASR配置
    asr_language: str = Field(default="zh-CN")
    asr_model: str = Field(default="streaming")
    enable_vad: bool = Field(default=True)
    vad_silence_duration: float = Field(default=1.0)
    
    # TTS配置
    tts_voice: str = Field(default="BV001_streaming")
    tts_speed: float = Field(default=1.0)
    tts_volume: float = Field(default=1.0)
    
    # 对话配置
    bot_name: str = Field(default="小助手")
    system_role: str = Field(default="你是一个智能助手，请用简洁友好的语言回答用户问题。")
    max_history_turns: int = Field(default=10)
    
    # 连接配置
    connection_timeout: float = Field(default=30.0)
    heartbeat_interval: float = Field(default=30.0)
    max_reconnect_attempts: int = Field(default=3)
    
    # 调试配置
    debug_mode: bool = Field(default=False)
    log_audio_data: bool = Field(default=False)


class VolcEngineRealtimeContext(HandlerContext):
    """火山引擎实时对话上下文"""
    
    def __init__(self, session_id: str):
        super().__init__(session_id=session_id)
        self.client: Optional[VolcEngineRealtimeClient] = None
        self.audio_device_manager: Optional[AudioDeviceManager] = None
        self.audio_config: Optional[AudioConfig] = None
        self.audio_recorder: Optional[AudioRecorder] = None
        self.is_connected: bool = False
        self.is_session_active: bool = False
        self.audio_input_queue: queue.Queue = queue.Queue()
        self.audio_output_queue: queue.Queue = queue.Queue()
        self.text_output_queue: queue.Queue = queue.Queue()
        self.processing_thread: Optional[threading.Thread] = None
        self.output_thread: Optional[threading.Thread] = None
        self.stop_event: threading.Event = threading.Event()
        self.config: Optional[VolcEngineRealtimeConfig] = None
        self.current_recording_file: Optional[str] = None
        self.chat_buffer: str = ""  # 聊天文本缓冲区
        self.chat_session_active: bool = False  # 聊天session状态


class HandlerVolcEngineRealtime(HandlerBase, ABC):
    """火山引擎实时对话处理器"""
    
    def __init__(self):
        super().__init__()
        self.logger = logger
        self.config: Optional[VolcEngineRealtimeConfig] = None
        self._current_context: Optional[VolcEngineRealtimeContext] = None
        
        # 在Handler级别管理dialog_id，跨session保持
        self.current_dialog_id: Optional[str] = None
        self.dialog_history: Dict[str, Any] = {}  # 可选：保存对话历史
        
        self.logger.info("[VOLCENGINE_INIT] 火山引擎实时对话处理器开始初始化")
        self.logger.info("[VOLCENGINE_INIT] 火山引擎实时对话处理器初始化完成")
    
    def get_handler_info(self) -> HandlerBaseInfo:
        """获取处理器信息"""
        return HandlerBaseInfo(
            config_model=VolcEngineRealtimeConfig,
            load_priority=0
        )
    
    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[HandlerBaseConfigModel] = None):
        """加载处理器"""
        self.logger.info("[VOLCENGINE_LOAD] 开始加载火山引擎实时对话处理器")
        self.logger.info(f"[VOLCENGINE_LOAD] 引擎配置类型: {type(engine_config)}")
        self.logger.info(f"[VOLCENGINE_LOAD] 处理器配置类型: {type(handler_config)}")
        
        if handler_config and not isinstance(handler_config, VolcEngineRealtimeConfig):
            self.logger.error(f"[VOLCENGINE_LOAD] 配置类型错误: {type(handler_config)}")
            raise ValueError("Invalid config type for VolcEngine realtime handler")
        
        # 验证必要的配置
        if handler_config:
            config = handler_config
            self.logger.info(f"[VOLCENGINE_LOAD] 验证配置 - app_id存在: {bool(config.app_id)}, access_key存在: {bool(config.access_key)}")
            if not config.app_id or not config.access_key:
                self.logger.error("[VOLCENGINE_LOAD] app_id或access_key缺失")
                raise ValueError("app_id and access_key are required for VolcEngine realtime handler")
        else:
            self.logger.warning("[VOLCENGINE_LOAD] 未提供处理器配置")
        
        self.logger.info("[VOLCENGINE_LOAD] 火山引擎实时对话处理器加载完成")
    
    def create_context(self, session_context: SessionContext, 
                      handler_config: Optional[HandlerBaseConfigModel] = None) -> HandlerContext:
        """创建处理器上下文"""
        session_id = session_context.session_info.session_id
        self.logger.info(f"[VOLCENGINE_CONTEXT] 开始创建上下文，会话ID: {session_id}")
        context = VolcEngineRealtimeContext(session_id=session_id)
        
        if handler_config and isinstance(handler_config, VolcEngineRealtimeConfig):
            self.logger.info(f"[VOLCENGINE_CONTEXT] 配置有效，开始初始化组件")
            context.config = handler_config
            
            # 初始化音频配置
            self.logger.info(f"[VOLCENGINE_CONTEXT] 初始化音频配置 - 采样率: {handler_config.sample_rate}, 通道数: {handler_config.channels}")
            import pyaudio
            audio_format = pyaudio.paInt16 if handler_config.bits_per_sample == 16 else pyaudio.paInt32
            audio_config = AudioConfig(
                sample_rate=handler_config.sample_rate,
                channels=handler_config.channels,
                chunk_size=handler_config.chunk_size,
                format=audio_format
            )
            
            # 初始化音频处理器
            self.logger.info(f"[VOLCENGINE_CONTEXT] 初始化音频处理器")
            context.audio_device_manager = AudioDeviceManager(audio_config)
            context.audio_config = audio_config
            
            # 初始化音频播放器（移除录制器，因为使用音频文件模式）
            self.logger.info(f"[VOLCENGINE_CONTEXT] 初始化音频播放器（音频文件模式）")
            context.audio_recorder = None  # 不使用录制器
            
            context.audio_player = AudioPlayer(
                sample_rate=handler_config.sample_rate,
                channels=handler_config.channels
            )
            
            # 初始化火山引擎客户端
            self.logger.info(f"[VOLCENGINE_CONTEXT] 初始化火山引擎客户端 - URL: {handler_config.ws_url}")
            ws_config = {
                'base_url': handler_config.ws_url,
                'headers': {
                    'X-Api-App-ID': handler_config.app_id,
                    'X-Api-Access-Key': handler_config.access_key,
                    'X-Api-Resource-Id': handler_config.resource_id,
                    'X-Api-App-Key': handler_config.app_key,
                    'X-Api-Connect-Id': str(uuid.uuid4()),
                }
            }
            session_config = {
                'model': handler_config.asr_model,
                'voice': handler_config.tts_voice,
                'language': handler_config.asr_language
            }
            context.client = VolcEngineRealtimeClient(ws_config, context.session_id)
            self.logger.info(f"[VOLCENGINE_CONTEXT] 火山引擎客户端初始化完成")
        else:
            self.logger.warning(f"[VOLCENGINE_CONTEXT] 配置无效或缺失: {type(handler_config)}")
        
        self.logger.info(f"[VOLCENGINE_CONTEXT] 创建火山引擎实时对话上下文完成: {session_context.session_info.session_id}")
        return context
    
    def start_context(self, session_context: SessionContext, handler_context: HandlerContext):
        """启动上下文"""
        session_id = session_context.session_info.session_id
        self.logger.info(f"[VOLCENGINE_START] 开始启动上下文，会话ID: {session_id}")
        
        if not isinstance(handler_context, VolcEngineRealtimeContext):
            self.logger.error(f"[VOLCENGINE_START] 上下文类型错误: {type(handler_context)}")
            raise ValueError("Invalid context type for VolcEngine realtime handler")
        
        context = handler_context
        self.logger.info(f"[VOLCENGINE_START] 上下文类型验证通过")
        
        # 保存当前上下文引用，供回调函数使用
        self._current_context = context
        self.logger.info(f"[VOLCENGINE_START] 当前上下文引用已保存")
        
        # 启动处理线程
        self.logger.info(f"[VOLCENGINE_START] 创建处理线程")
        context.processing_thread = threading.Thread(
            target=self._processing_loop,
            args=(context,),
            daemon=True
        )
        context.processing_thread.start()
        self.logger.info(f"[VOLCENGINE_START] 处理线程已启动，线程ID: {context.processing_thread.ident}")
        
        # 启动输出线程
        self.logger.info(f"[VOLCENGINE_START] 创建输出线程")
        context.output_thread = threading.Thread(
            target=self._output_loop,
            args=(context,),
            daemon=True
        )
        context.output_thread.start()
        self.logger.info(f"[VOLCENGINE_START] 输出线程已启动，线程ID: {context.output_thread.ident}")
        
        self.logger.info(f"[VOLCENGINE_START] 启动火山引擎实时对话上下文完成: {session_id}")
    
    def get_handler_detail(self, session_context: SessionContext, 
                          context: HandlerContext) -> HandlerDetail:
        """获取处理器详情"""
        # 定义输入数据格式
        inputs = {
            ChatDataType.MIC_AUDIO: HandlerDataInfo(
                type=ChatDataType.MIC_AUDIO,
            )
        }
        
        # 定义输出数据格式
        text_definition = DataBundleDefinition()
        text_definition.add_entry(DataBundleEntry.create_text_entry("avatar_text"))
        
        audio_definition = DataBundleDefinition()
        audio_definition.add_entry(DataBundleEntry.create_audio_entry(
            "avatar_audio", 
            channel_num=1, 
            sample_rate=16000
        ))
        
        outputs = {
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(
                type=ChatDataType.AVATAR_TEXT,
                definition=text_definition,
            ),
            ChatDataType.AVATAR_AUDIO: HandlerDataInfo(
                type=ChatDataType.AVATAR_AUDIO,
                definition=audio_definition,
            )
        }
        
        return HandlerDetail(
            inputs=inputs,
            outputs=outputs,
        )
    
    async def handle(self, context: HandlerContext, inputs: ChatData,
                    output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        """处理输入数据"""
        # 初始化计数器
        if not hasattr(self, '_handle_call_count'):
            self._handle_call_count = 0
        
        if not isinstance(context, VolcEngineRealtimeContext):
            raise ValueError("Invalid context type for VolcEngine realtime handler")
        
        # 检查录音信号元数据 - 修改为使用音频文件而非录音
        recording_signal = None
        if hasattr(inputs.data, 'get_meta'):
            recording_signal = inputs.data.get_meta('recording_signal')
            if recording_signal:
                self.logger.info(f"[VOLCENGINE_HANDLER] 检测到录音信号: {recording_signal}")
                
                # 处理录音开始信号 - 跳过实际录音，仅记录日志
                if recording_signal == 'start':
                    self.logger.info(f"[VOLCENGINE_HANDLER] 收到录音开始信号，但当前配置为使用音频文件模式")
                
                # 处理录音停止信号 - 使用预设的音频文件
                elif recording_signal == 'stop':
                    self.logger.info(f"[VOLCENGINE_HANDLER] 收到录音停止信号，使用音频文件模式")
                    
                    # 使用预设的音频文件路径（可以从配置中获取或使用默认路径）
                    temp_file_path = r"D:\Projects\TA\OpenAvatarChat\src\handlers\realtime\volcengine\volcengine_real_time_official\python3.7\whoareyou.wav"  # 使用实际的音频文件
                    context.current_recording_file = temp_file_path
                    self.logger.info(f"[VOLCENGINE_HANDLER] 使用音频文件: {temp_file_path}")
                    
                    # 发送音频文件到服务器并等待返回
                    self.logger.info(f"[VOLCENGINE_HANDLER] 开始发送音频文件到服务器: {temp_file_path}")
                    await self._send_temp_file_and_wait_response(context, temp_file_path)
                    
                    # 处理接收到的响应数据
                    self._process_received_responses(context, output_definitions)
                    
                    # 设置停止事件以触发FinishSession
                    self.logger.info(f"[VOLCENGINE_HANDLER] 设置停止事件，准备发送FinishSession")
                    context.stop_event.set()
        
        # 处理音频输入
        if inputs.type == ChatDataType.MIC_AUDIO:
            try:
                self._handle_call_count += 1
                # 获取音频数据
                audio_data = inputs.data.get_main_data()
                
                # 检查音频数据有效性，过滤掉过小的数据包
                if len(audio_data) < 10:  # 过滤掉小于10字节的无效数据
                    if self._handle_call_count % 1000 == 0:  # 减少无效数据的日志频率
                        self.logger.warning(f"[VOLCENGINE_HANDLER] 跳过无效音频数据: {len(audio_data)} 字节 (总计跳过: {self._handle_call_count}次)")
                    return
                
                # 处理音频数据 - 新的AudioDeviceManager不需要预处理
                # 直接将音频数据放入队列
                context.audio_input_queue.put(audio_data)
                
                # 只在首次或每100次调用时记录日志
                if self._handle_call_count == 1 or self._handle_call_count % 100 == 0:
                    queue_size = context.audio_input_queue.qsize()
                    self.logger.info(f"[VOLCENGINE_HANDLER] 已处理{self._handle_call_count}个有效音频包，队列大小: {queue_size}")
                
                # 调试模式下的详细日志（更少频率）
                if (context.config and context.config.debug_mode and 
                    context.config.log_audio_data and self._handle_call_count % 200 == 0):
                    self.logger.debug(f"[VOLCENGINE_HANDLER] 调试模式 - 音频数据详情: {len(audio_data)} 字节")
                    
            except Exception as e:
                self.logger.error(f"[VOLCENGINE_HANDLER] 处理音频输入时出错: {e}")
                import traceback
                self.logger.error(f"[VOLCENGINE_HANDLER] 错误堆栈: {traceback.format_exc()}")
        else:
            self.logger.info(f"[VOLCENGINE_HANDLER] 收到非音频数据类型: {inputs.type}")
    
    def destroy_context(self, context: HandlerContext):
        """销毁上下文"""
        if not isinstance(context, VolcEngineRealtimeContext):
            return
        
        self.logger.info(f"开始销毁火山引擎实时对话上下文: {context.session_id}")
        
        # 停止处理线程
        context.stop_event.set()
        
        # 清理音频输入队列
        if hasattr(context, 'audio_input_queue') and context.audio_input_queue:
            queue_size = context.audio_input_queue.qsize()
            if queue_size > 0:
                self.logger.info(f"清理音频队列中的 {queue_size} 个数据包")
                # 清空队列
                while not context.audio_input_queue.empty():
                    try:
                        context.audio_input_queue.get_nowait()
                    except queue.Empty:
                        break
                self.logger.info("音频队列清理完成")
        
        # 注释掉断开连接逻辑，保持WebSocket连接以接收服务器响应
        # if context.client and context.is_connected:
        #     try:
        #         asyncio.run(context.client.disconnect())
        #     except Exception as e:
        #         self.logger.error(f"断开连接时出错: {e}")
        self.logger.info(f"[VOLCENGINE_DESTROY] 保持WebSocket连接，跳过断开逻辑: {context.session_id}")
        
        # 等待线程结束
        if context.processing_thread and context.processing_thread.is_alive():
            self.logger.info("等待处理线程结束...")
            context.processing_thread.join(timeout=5.0)
            if context.processing_thread.is_alive():
                self.logger.warning("处理线程未能在5秒内结束")
        
        if context.output_thread and context.output_thread.is_alive():
            self.logger.info("等待输出线程结束...")
            context.output_thread.join(timeout=5.0)
            if context.output_thread.is_alive():
                self.logger.warning("输出线程未能在5秒内结束")
        
        self.logger.info(f"火山引擎实时对话上下文销毁完成: {context.session_id}")
    
    def _processing_loop(self, context: VolcEngineRealtimeContext):
        """音频处理循环"""
        self.logger.info(f"[VOLCENGINE_PROCESSING] 处理循环开始，会话ID: {context.session_id}")
        try:
            # 建立连接
            self.logger.info(f"[VOLCENGINE_PROCESSING] 开始异步连接和处理")
            asyncio.run(self._connect_and_process(context))
            self.logger.info(f"[VOLCENGINE_PROCESSING] 异步连接和处理完成")
        except Exception as e:
            self.logger.error(f"[VOLCENGINE_PROCESSING] 处理循环出错: {e}")
            import traceback
            self.logger.error(f"[VOLCENGINE_PROCESSING] 错误堆栈: {traceback.format_exc()}")
    
    async def _connect_and_process(self, context: VolcEngineRealtimeContext):
        """连接并处理音频数据"""
        self.logger.info(f"[VOLCENGINE_CONNECT] 开始连接和处理流程: {context.session_id}")
        try:
            self.logger.info(f"[VOLCENGINE_CONNECT] 检查客户端状态: {context.client is not None}")
            if context.client is None:
                self.logger.error(f"[VOLCENGINE_CONNECT] 客户端未初始化")
                return
            
            # 火山引擎客户端不使用回调模式，而是使用直接接收模式
            self.logger.info(f"[VOLCENGINE_CONNECT] 火山引擎客户端已准备就绪: {context.session_id}")
            
            # 连接到火山引擎
            self.logger.info(f"[VOLCENGINE_CONNECT] 开始WebSocket连接: {context.session_id}")
            await context.client.start_connection()
            context.is_connected = True
            self.logger.info(f"[VOLCENGINE_CONNECT] 火山引擎WebSocket连接成功: {context.session_id}")
            
            # 启动会话
            self.logger.info(f"[VOLCENGINE_CONNECT] 启动火山引擎会话: {context.session_id}")
            
            # 构建会话请求参数
            if self.current_dialog_id:
                dialog = {
                    "bot_name": context.config.bot_name,
                    "system_role": context.config.system_role,
                    "speaking_style": "你的说话风格简洁明了，语速适中，语调自然。",
                    "dialog_id": self.current_dialog_id,
                    "location": {
                        "city": "北京",
                    },
                    "extra": {
                        "strict_audit": False,
                        "audit_response": "支持客户自定义安全审核回复话术。"
                    }
                }
            else:
                dialog = {
                    "bot_name": context.config.bot_name,
                    "system_role": context.config.system_role,
                    "speaking_style": "你的说话风格简洁明了，语速适中，语调自然。",
                    "location": {
                        "city": "北京",
                    },
                    "extra": {
                        "strict_audit": False,
                        "audit_response": "支持客户自定义安全审核回复话术。"
                    }
                }


            start_session_req = {
                "asr": {
                    "extra": {
                        "end_smooth_window_ms": 1500,
                    },
                },
                "tts": {
                    "speaker": context.config.tts_voice,
                    "audio_config": {
                        "channel": context.config.channels,
                        "format": context.config.audio_format,
                        "sample_rate": context.config.sample_rate
                    },
                },
                "dialog": dialog
            }
            
            await context.client.start_session(start_session_req)
            context.is_session_active = True
            self.logger.info(f"[VOLCENGINE_CONNECT] 火山引擎会话启动成功: {context.session_id}")
            
            # 创建音频发送和响应接收任务
            self.logger.info(f"[VOLCENGINE_CONNECT] 开始音频处理和响应接收: {context.session_id}")
            
            # 创建并发任务
            audio_task = asyncio.create_task(self._handle_audio_input(context))
            response_task = asyncio.create_task(self._handle_server_responses(context))
            
            # 等待任务完成或停止事件
            try:
                await asyncio.gather(audio_task, response_task, return_exceptions=True)
            except Exception as e:
                self.logger.error(f"[VOLCENGINE_CONNECT] 处理任务出错: {e}")
                # 取消未完成的任务
                if not audio_task.done():
                    audio_task.cancel()
                if not response_task.done():
                    response_task.cancel()
            
        except Exception as e:
            self.logger.error(f"[VOLCENGINE_CONNECT] 连接和处理过程出错: {e}")
            import traceback
            self.logger.error(f"[VOLCENGINE_CONNECT] 连接错误堆栈: {traceback.format_exc()}")
        finally:
            self.logger.info(f"[VOLCENGINE_CONNECT] 开始清理连接资源: {context.session_id}")
            
            # 等待可能的服务器响应
            self.logger.info(f"[VOLCENGINE_CONNECT] 等待服务器响应完成: {context.session_id}")
            await asyncio.sleep(2.0)  # 等待2秒让服务器有时间返回响应
            
            # 根据火山引擎实际行为，服务器会在FinishSession后主动关闭连接
            # 因此我们需要正常结束会话，为下次连接做准备
            if context.is_session_active:
                try:
                    self.logger.info(f"[VOLCENGINE_CONNECT] 开始结束火山引擎会话: {context.session_id}")
                    await context.client.finish_session()
                    context.is_session_active = False
                    self.logger.info(f"[VOLCENGINE_CONNECT] 火山引擎会话已结束，服务器将关闭连接: {context.session_id}")
                except Exception as e:
                    self.logger.error(f"[VOLCENGINE_CONNECT] 结束会话时出错: {e}")
            
            # 等待服务器关闭连接，然后清理客户端状态
            await asyncio.sleep(1.0)
            
            if context.is_connected:
                try:
                    self.logger.info(f"[VOLCENGINE_CONNECT] 清理WebSocket连接状态: {context.session_id}")
                    await context.client.finish_connection()
                    context.is_connected = False
                    self.logger.info(f"[VOLCENGINE_CONNECT] WebSocket连接状态已清理: {context.session_id}")
                except Exception as e:
                    self.logger.error(f"[VOLCENGINE_CONNECT] 清理连接时出错: {e}")
            
            self.logger.info(f"[VOLCENGINE_CONNECT] 会话处理完成，准备下次连接: {context.session_id}")
    
    def _output_loop(self, context: VolcEngineRealtimeContext):
        """输出处理循环"""
        self.logger.info(f"[VOLCENGINE_OUTPUT] 启动输出循环: {context.session_id}")
        output_count = 0
        last_log_count = 0
        
        while not context.stop_event.is_set():
            try:
                # 处理文本输出
                if not context.text_output_queue.empty():
                    text_data = context.text_output_queue.get_nowait()
                    output_count += 1
                    
                    # 记录文本输出处理
                    self.logger.info(f"[VOLCENGINE_OUTPUT] 处理文本输出#{output_count}: {text_data[:100]}{'...' if len(text_data) > 100 else ''}")
                    
                    # 创建文本输出数据
                    from chat_engine.data_models.chat_data.chat_data_model import ChatData
                    from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
                    
                    text_definition = DataBundleDefinition()
                    text_definition.add_entry(DataBundleEntry.create_text_entry("assistant_text"))
                    text_bundle = DataBundle(text_definition)
                    text_bundle.set_data("assistant_text", text_data)
                    
                    text_chat_data = ChatData(
                        type=ChatDataType.ASSISTANT_TEXT,
                        data=text_bundle
                    )
                    
                    # 发送到输出队列
                    self.logger.info(f"[VOLCENGINE_OUTPUT] 将文本数据发送到输出队列，准备交给LAM模块处理")
                    context.output_queue.put(text_chat_data)
                    self.logger.info(f"[VOLCENGINE_OUTPUT] 文本数据已成功放入输出队列，队列大小: {context.output_queue.qsize()}")
                    
                    if context.config and context.config.debug_mode:
                        self.logger.debug(f"[VOLCENGINE_OUTPUT] 调试模式 - 输出文本: {text_data}")
                
                # 处理音频输出
                if not context.audio_output_queue.empty():
                    audio_data = context.audio_output_queue.get_nowait()
                    output_count += 1
                    
                    # 记录音频输出处理
                    if output_count - last_log_count >= 10:
                        self.logger.info(f"[VOLCENGINE_OUTPUT] 音频输出处理中，已处理{output_count}个包")
                        last_log_count = output_count
                    
                    # 创建音频输出数据
                    from chat_engine.data_models.chat_data.chat_data_model import ChatData
                    from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
                    
                    audio_definition = DataBundleDefinition()
                    audio_definition.add_entry(DataBundleEntry.create_audio_entry(
                        "assistant_audio", 
                        channel_num=1, 
                        sample_rate=context.config.sample_rate if context.config else 16000
                    ))
                    audio_bundle = DataBundle(audio_definition)
                    audio_bundle.set_data("assistant_audio", audio_data)
                    
                    audio_chat_data = ChatData(
                        type=ChatDataType.ASSISTANT_AUDIO,
                        data=audio_bundle
                    )
                    
                    # 发送到输出队列
                    if output_count % 5 == 1:  # 每5个包记录一次详细日志
                        self.logger.info(f"[VOLCENGINE_OUTPUT] 将音频数据({len(audio_data)}字节)发送到输出队列，准备播放")
                    context.output_queue.put(audio_chat_data)
                    if output_count % 5 == 1:
                        self.logger.info(f"[VOLCENGINE_OUTPUT] 音频数据已成功放入输出队列，队列大小: {context.output_queue.qsize()}")
                    
                    if context.config and context.config.debug_mode and output_count % 20 == 0:
                        self.logger.debug(f"[VOLCENGINE_OUTPUT] 调试模式 - 输出音频: {len(audio_data)} 字节")
                
                time.sleep(0.01)  # 避免CPU占用过高
                
            except Exception as e:
                self.logger.error(f"[VOLCENGINE_OUTPUT] 输出处理出错: {e}")
                import traceback
                self.logger.error(f"[VOLCENGINE_OUTPUT] 输出错误堆栈: {traceback.format_exc()}")
                break
        
        self.logger.info(f"[VOLCENGINE_OUTPUT] 输出循环结束: {context.session_id}")
    
    async def _on_message_received(self, message: Dict[str, Any]):
        """处理接收到的文本消息"""
        try:
            # 识别消息事件类型
            event_type = message.get('event', 'unknown')
            self.logger.info(f"[VOLCENGINE_CALLBACK] 收到消息回调，事件类型: {event_type}")
            
            # 提取文本内容
            text_content = message.get('text') or message.get('message', '')
            if text_content:
                self.logger.info(f"[VOLCENGINE_CALLBACK] 提取到文本内容: {text_content[:100]}{'...' if len(text_content) > 100 else ''}")
                
                # 将文本放入输出队列
                if hasattr(self, '_current_context'):
                    self._current_context.text_output_queue.put(text_content)
                    queue_size = self._current_context.text_output_queue.qsize()
                    self.logger.info(f"[VOLCENGINE_CALLBACK] 文本已放入输出队列，当前队列大小: {queue_size}")
                    
                    # 根据事件类型记录特定信息
                    if event_type == 'asr':
                        self.logger.info(f"[VOLCENGINE_CALLBACK] ASR识别结果已处理，准备进行对话生成")
                    elif event_type == 'chat':
                        self.logger.info(f"[VOLCENGINE_CALLBACK] CHAT对话结果已处理，准备交给LAM和播放模块")
                else:
                    self.logger.warning(f"[VOLCENGINE_CALLBACK] 当前上下文不存在，无法处理文本消息")
            else:
                self.logger.warning(f"[VOLCENGINE_CALLBACK] 消息中未找到文本内容，完整消息: {message}")
                    
        except Exception as e:
            self.logger.error(f"[VOLCENGINE_CALLBACK] 处理文本消息出错: {e}")
            import traceback
            self.logger.error(f"[VOLCENGINE_CALLBACK] 文本消息错误堆栈: {traceback.format_exc()}")
    
    async def _on_audio_received(self, audio_data: bytes):
        """处理接收到的音频数据"""
        try:
            # 将音频数据放入输出队列
            if hasattr(self, '_current_context'):
                self._current_context.audio_output_queue.put(audio_data)
                
                # 初始化计数器
                if not hasattr(self, '_audio_callback_count'):
                    self._audio_callback_count = 0
                    self.logger.info(f"[VOLCENGINE_CALLBACK] 开始接收TTS音频数据流")
                    
                self._audio_callback_count += 1
                
                # 记录音频接收进度
                if self._audio_callback_count == 1:
                    self.logger.info(f"[VOLCENGINE_CALLBACK] 收到第一个TTS音频包，长度: {len(audio_data)} 字节")
                elif self._audio_callback_count % 50 == 0:
                    queue_size = self._current_context.audio_output_queue.qsize()
                    self.logger.info(f"[VOLCENGINE_CALLBACK] 已接收{self._audio_callback_count}个TTS音频包，当前包长度: {len(audio_data)} 字节，队列大小: {queue_size}")
                
                # 记录音频数据准备播放
                if self._audio_callback_count == 1:
                    self.logger.info(f"[VOLCENGINE_CALLBACK] TTS音频数据已放入播放队列，准备开始播放和LAM处理")
            else:
                self.logger.warning(f"[VOLCENGINE_CALLBACK] 当前上下文不存在，无法处理TTS音频数据")
        except Exception as e:
            self.logger.error(f"[VOLCENGINE_CALLBACK] 处理TTS音频数据出错: {e}")
            import traceback
            self.logger.error(f"[VOLCENGINE_CALLBACK] TTS音频数据错误堆栈: {traceback.format_exc()}")
    
    async def _send_temp_file_and_wait_response(self, context: VolcEngineRealtimeContext, temp_file_path: str):
        """发送临时文件到服务器并等待响应"""
        try:
            self.logger.info(f"[VOLCENGINE_FILE_SEND] 开始发送临时文件: {temp_file_path}")
            
            # 创建一个新的客户端连接用于发送文件
            file_client_config = {
                'ws_connect_config': {
                    'base_url': "wss://openspeech.bytedance.com/api/v3/realtime/dialogue", # 临时使用固定连接，等代码全调试通顺后再使用conifg文件，下同
                    'headers': {
                        'X-Api-App-ID': "5385188715",
                        'X-Api-Access-Key': "7Mb2pErVhX_z2bpEjWlYJa6N9yW2RhmS",
                        'X-Api-Resource-Id': "volc.speech.dialog",
                        'X-Api-App-Key': "PlgvMymc7f3tQnJ6",
                        "X-Api-Connect-Id": str(uuid.uuid4()),
                    }
                },
                'start_session_req': {
                    "asr": {
                        "extra": {
                            "end_smooth_window_ms": 1500,
                        },
                    },
                    "tts": {
                        "speaker": "zh_male_yunzhou_jupiter_bigtts",
                        "audio_config": {
                            "channel": 1,
                            "format": "pcm",
                            "sample_rate": 24000
                        },
                    }
                },
                'input_audio_config': {
                    "chunk": 3200,
                    "format": "pcm",
                    "channels": 1,
                    "sample_rate": 16000,
                    "bit_size": 16  # paInt16 对应的数值
                }
            }
            file_client = VolcEngineRealtimeClient(file_client_config, context.session_id)
            
            # 设置响应收集变量
            received_responses = []
            session_finished_event = asyncio.Event()
            
            # 连接到服务器
            self.logger.info(f"[VOLCENGINE_FILE_SEND] 连接到火山引擎服务器")
            await file_client.start_connection()
            
            # 启动会话
            self.logger.info(f"[VOLCENGINE_FILE_SEND] 启动会话")
            
            # 构建会话请求参数
            start_session_req = file_client_config["start_session_req"]
            
            await file_client.start_session(start_session_req)
            self.logger.info(f"协议启动完毕")
            # 读取并发送音频文件
            import wave
            
            try:
                with wave.open(temp_file_path, 'rb') as wav_file:
                    chunk_size = file_client_config["input_audio_config"]["chunk"]
                    self.logger.info(f"开始处理音频文件: {temp_file_path}")
                    while True:
                        audio_data = wav_file.readframes(chunk_size)
                        if not audio_data:
                            break
                        
                        await file_client.send_audio_data(audio_data)
                        # sent_frames += frames_to_read
                        
                        # 等待20ms模拟实时发送
                        # await asyncio.sleep(0.02)
                        
                        # 检查是否收到完整响应
                        # if session_finished_event.is_set():
                        #     self.logger.info(f"[VOLCENGINE_FILE_SEND] 收到完整响应，停止发送音频")
                        #     break
                    
                    self.logger.info(f"[VOLCENGINE_FILE_SEND] 音频文件发送完成")
                    
            except Exception as e:
                self.logger.error(f"[VOLCENGINE_FILE_SEND] 发送音频文件时出错: {e}")
                return
            
            # 使用事件机制等待服务器响应
            self.logger.info(f"[VOLCENGINE_FILE_SEND] 开始等待服务器响应...")
            max_wait_time = 30.0  # 最大等待30秒
            
            # 创建静音数据发送任务（模拟官方示例的process_silence_audio）
            async def send_silence_audio():
                """持续发送静音音频数据，防止连接超时"""
                silence_data = b'\x00' * 320  # 320字节的静音数据
                try:
                    while not session_finished_event.is_set():
                        await file_client.send_audio_data(silence_data)
                        await asyncio.sleep(0.01)  # 每10ms发送一次静音数据
                except Exception as e:
                    self.logger.warning(f"[VOLCENGINE_FILE_SEND] 发送静音数据时出错: {e}")
            
            # 启动静音数据发送任务
            silence_task = asyncio.create_task(send_silence_audio())
            
            # 创建响应处理任务
            async def handle_responses():
                """处理服务器响应"""
                try:
                    while not session_finished_event.is_set():
                        try:
                            response = await asyncio.wait_for(file_client.receive_server_response(), timeout=1.0)
                            received_responses.append(response)
                            if response['event'] != 352:
                                self.logger.info(f"[VOLCENGINE_FILE_SEND] 收到服务器响应: {response}")
                            else:
                                self.logger.info(f"[VOLCENGINE_FILE_SEND] 收到服务器音频响应，前100字符为: {response['payload_msg'][0:100]}")
                            
                            # 解析和处理响应数据
                            await self._parse_and_handle_response(context, response)
                            
                            # 检查是否包含会话结束事件
                            if isinstance(response, dict):
                                event = response.get('event')
                                if event in [152, 153, 359]:  # 会话结束事件
                                    self.logger.info(f"[VOLCENGINE_FILE_SEND] 收到会话结束事件: {event}")
                                    session_finished_event.set()
                                    break
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            self.logger.error(f"[VOLCENGINE_FILE_SEND] 接收响应时出错: {e}")
                            break
                except Exception as e:
                    self.logger.error(f"[VOLCENGINE_FILE_SEND] 响应处理任务出错: {e}")
            
            # 启动响应处理任务
            response_task = asyncio.create_task(handle_responses())
            
            try:
                # 等待会话结束事件或超时
                await asyncio.wait_for(session_finished_event.wait(), timeout=max_wait_time)
                self.logger.info(f"[VOLCENGINE_FILE_SEND] 收到会话结束信号")
            except asyncio.TimeoutError:
                self.logger.warning(f"[VOLCENGINE_FILE_SEND] 等待服务器响应超时({max_wait_time}秒)")
            finally:
                # 设置事件以停止所有任务
                session_finished_event.set()
                
                # 取消静音数据发送任务
                if silence_task and not silence_task.done():
                    silence_task.cancel()
                    try:
                        await silence_task
                    except asyncio.CancelledError:
                        pass
                
                # 取消响应处理任务
                if response_task and not response_task.done():
                    response_task.cancel()
                    try:
                        await response_task
                    except asyncio.CancelledError:
                        pass
            
            # 记录收到的响应
            if received_responses:
                self.logger.info(f"[VOLCENGINE_FILE_SEND] 共收到 {len(received_responses)} 个服务器响应")
                for i, response in enumerate(received_responses):
                    if response["event"]!=352:
                        self.logger.info(f"[VOLCENGINE_FILE_SEND] 响应 #{i+1}: {response}")
                    else:
                        self.logger.info(f"[VOLCENGINE_FILE_SEND] 音频响应 #{i+1}，前百字符为: {response['payload_msg'][0:100]}")
            else:
                self.logger.warning(f"[VOLCENGINE_FILE_SEND] 未收到任何服务器响应")
            
            # 结束会话
            self.logger.info(f"[VOLCENGINE_FILE_SEND] 结束会话")
            await file_client.finish_session()
            
            # 等待会话结束
            await asyncio.sleep(1.0)
            
            # 关闭连接
            self.logger.info(f"[VOLCENGINE_FILE_SEND] 关闭连接")
            await file_client.finish_connection()
            
            self.logger.info(f"[VOLCENGINE_FILE_SEND] 临时文件发送和响应处理完成")
            
        except Exception as e:
            self.logger.error(f"[VOLCENGINE_FILE_SEND] 发送临时文件时出错: {e}")
            import traceback
            self.logger.error(f"[VOLCENGINE_FILE_SEND] 错误堆栈: {traceback.format_exc()}")
    
    async def _handle_audio_input(self, context: VolcEngineRealtimeContext):
        """处理音频输入的异步任务"""
        self.logger.info(f"[VOLCENGINE_AUDIO] 开始音频输入处理循环: {context.session_id}")
        audio_count = 0
        
        while not context.stop_event.is_set():
            try:
                # 从队列获取音频数据
                audio_data = context.audio_input_queue.get(timeout=0.1)
                audio_count += 1
                self.logger.info(f"[VOLCENGINE_AUDIO] 获取到音频输入数据#{audio_count}，长度: {len(audio_data)} 字节")
                
                # 发送音频数据
                await context.client.send_audio_data(audio_data)
                self.logger.info(f"[VOLCENGINE_AUDIO] 音频数据#{audio_count}已发送到火山引擎，长度: {len(audio_data)} 字节")
                
            except queue.Empty:
                # 短暂等待避免CPU占用过高
                await asyncio.sleep(0.01)
                continue
            except Exception as e:
                self.logger.error(f"[VOLCENGINE_AUDIO] 发送音频数据时出错: {e}")
                import traceback
                self.logger.error(f"[VOLCENGINE_AUDIO] 发送音频错误堆栈: {traceback.format_exc()}")
                break
        
        self.logger.info(f"[VOLCENGINE_AUDIO] 音频输入处理循环结束: {context.session_id}")
    
    async def _parse_and_handle_response(self, context: VolcEngineRealtimeContext, response: Dict[str, Any]):
        """解析并处理服务器响应"""
        try:
            if not response:
                return
                
            message_type = response.get('message_type')
            event = response.get('event')
            payload_msg = response.get('payload_msg')
            
            self.logger.info(f"[VOLCENGINE_PARSE] 解析响应 - 类型: {message_type}, 事件: {event}")
            
            # 直接根据事件代码处理，不再区分message_type
            if event == 451:  # ASR事件
                await self._handle_asr_response(context, payload_msg)
            elif event == 350:  # TTS配置事件
                await self._handle_tts_response(context, payload_msg)
            elif event == 352:  # TTS音频数据事件（重要！）
                if isinstance(payload_msg, bytes):
                    await self._handle_audio_data(context, payload_msg)
                elif isinstance(payload_msg, dict) and 'audio' in payload_msg:
                    audio_data = payload_msg['audio']
                    if isinstance(audio_data, bytes):
                        await self._handle_audio_data(context, audio_data)
            elif event == 150:  # 会话开始事件
                self.logger.info(f"[VOLCENGINE_PARSE] 会话开始: {payload_msg}")
                await self._handle_session_started(context, payload_msg)
            elif event == 152:  # 会话结束事件
                self.logger.info(f"[VOLCENGINE_PARSE] 收到会话结束事件")
                self._handle_chat_session_end(context)  # 结束时输出缓冲区内容
            elif event == 153:  # 对话结束事件
                self.logger.info(f"[VOLCENGINE_PARSE] 收到对话结束事件")
                self._handle_chat_session_end(context)  # 结束时输出缓冲区内容
            elif event == 450:  # ASR信息事件（用于打断功能）
                self.logger.info(f"[VOLCENGINE_PARSE] ASR信息事件: {payload_msg}")
            elif event == 550:  # 聊天响应事件
                await self._handle_chat_response(context, payload_msg)
            else:
                # 处理没有事件代码的情况（如纯音频数据）
                if message_type == 'SERVER_ACK':
                    if isinstance(payload_msg, bytes):
                        # 直接处理二进制音频数据
                        await self._handle_audio_data(context, payload_msg)
                    elif isinstance(payload_msg, dict) and 'audio' in payload_msg:
                        # 处理包含音频字段的响应
                        audio_data = payload_msg['audio']
                        if isinstance(audio_data, bytes):
                            await self._handle_audio_data(context, audio_data)
                elif message_type == 'SERVER_ERROR_RESPONSE':
                    self.logger.error(f"[VOLCENGINE_PARSE] 服务器错误响应: {payload_msg}")
                else:
                    self.logger.info(f"[VOLCENGINE_PARSE] 收到未知事件: {event}, 消息类型: {message_type}")
                    
        except Exception as e:
            self.logger.error(f"[VOLCENGINE_PARSE] 解析响应时出错: {e}")
            import traceback
            self.logger.error(f"[VOLCENGINE_PARSE] 错误堆栈: {traceback.format_exc()}")
    
    async def _handle_asr_response(self, context: VolcEngineRealtimeContext, payload_msg: Any):
        """处理ASR（语音识别）响应"""
        try:
            if not isinstance(payload_msg, dict):
                self.logger.warning(f"[VOLCENGINE_ASR] ASR响应格式不正确: {type(payload_msg)}")
                return
            
            # 提取ASR结果
            results = payload_msg.get('results', [])
            if results:
                for result in results:
                    alternatives = result.get('alternatives', [])
                    if alternatives:
                        # 获取最佳识别结果
                        best_alternative = alternatives[0]
                        text = best_alternative.get('text', '')
                        confidence = best_alternative.get('confidence', 0.0)
                        
                        if text:
                            self.logger.info(f"[VOLCENGINE_ASR] 识别文本: '{text}' (置信度: {confidence})")
                            
                            # 将识别的文本放入输出队列
                            context.text_output_queue.put({
                                'type': 'asr_result',
                                'text': text,
                                'confidence': confidence,
                                'timestamp': time.time()
                            })
            
            # 记录其他信息
            asr_task_id = payload_msg.get('asr_task_id')
            is_final = payload_msg.get('is_final', False)
            round_id = payload_msg.get('round_id')
            
            self.logger.debug(f"[VOLCENGINE_ASR] ASR任务ID: {asr_task_id}, 是否最终结果: {is_final}, 轮次ID: {round_id}")
            
        except Exception as e:
            self.logger.error(f"[VOLCENGINE_ASR] 处理ASR响应时出错: {e}")
    
    async def _handle_tts_response(self, context: VolcEngineRealtimeContext, payload_msg: Any):
        """处理TTS（语音合成）响应"""
        try:
            if not isinstance(payload_msg, dict):
                self.logger.warning(f"[VOLCENGINE_TTS] TTS响应格式不正确: {type(payload_msg)}")
                return
            
            # 记录TTS响应信息
            tts_task_id = payload_msg.get('tts_task_id')
            tts_type = payload_msg.get('tts_type')
            
            self.logger.info(f"[VOLCENGINE_TTS] TTS响应 - 任务ID: {tts_task_id}, 类型: {tts_type}")
            
            # TTS响应通常不包含文本内容，主要是任务配置信息
            # 实际的音频数据会通过SERVER_ACK消息发送
            
        except Exception as e:
            self.logger.error(f"[VOLCENGINE_TTS] 处理TTS响应时出错: {e}")
    
    async def _handle_chat_response(self, context: VolcEngineRealtimeContext, payload_msg: Any):
        """处理聊天响应数据（事件550）- 基于session收集所有文本"""
        try:
            if not isinstance(payload_msg, dict):
                self.logger.warning(f"[VOLCENGINE_CHAT] 聊天响应格式不正确: {type(payload_msg)}")
                return
                
            content = payload_msg.get('content', '')
            
            # 初始化聊天缓冲区（如果不存在）
            if not hasattr(context, 'chat_buffer'):
                context.chat_buffer = ""
                context.chat_session_active = False
                
            # 如果内容为空，标记为session开始
            if content == '':
                if context.chat_session_active:
                    # 如果之前有session在进行，先输出之前的内容
                    self._flush_chat_buffer(context)
                
                # 开始新的聊天session
                context.chat_buffer = ""
                context.chat_session_active = True
                self.logger.info(f"[VOLCENGINE_CHAT] 开始新的聊天session")
                return
                
            # 如果session处于活跃状态，收集所有文本内容
            if context.chat_session_active:
                context.chat_buffer += content
                self.logger.info(f"[VOLCENGINE_CHAT] 收集聊天片段: '{content}', 当前缓冲区长度: {len(context.chat_buffer)}")
            else:
                # 如果没有活跃session，直接输出单个内容
                if content.strip():
                    context.text_output_queue.put(content.strip())
                    self.logger.info(f"[VOLCENGINE_CHAT] 直接输出聊天内容: '{content}'")
                    
        except Exception as e:
            self.logger.error(f"[VOLCENGINE_CHAT] 处理聊天响应时出错: {e}")
    
    def _flush_chat_buffer(self, context: VolcEngineRealtimeContext):
        """输出聊天缓冲区的内容"""
        try:
            if hasattr(context, 'chat_buffer') and context.chat_buffer.strip():
                complete_message = context.chat_buffer.strip()
                context.text_output_queue.put(complete_message)
                self.logger.info(f"[VOLCENGINE_CHAT] 输出完整聊天消息: '{complete_message}'")
                
            # 清空缓冲区并标记session结束
            context.chat_buffer = ""
            context.chat_session_active = False
            
        except Exception as e:
            self.logger.error(f"[VOLCENGINE_CHAT] 输出聊天缓冲区时出错: {e}")
    
    async def _handle_session_started(self, context: VolcEngineRealtimeContext, payload_msg: Any):
        """
        处理会话开始事件（事件150）
        """
        try:
            if isinstance(payload_msg, dict):
                # 在Handler级别保存dialog_id，而不是context级别
                dialog_id = payload_msg.get('dialog_id')
                if dialog_id:
                    self.current_dialog_id = dialog_id
                    self.logger.info(f"[VOLCENGINE_SESSION] 更新Handler级别dialog_id: {dialog_id}")
                    
                    # 可选：保存到对话历史
                    self.dialog_history[dialog_id] = {
                        'created_at': time.time(),
                        'session_id': context.session_id,
                        'payload': payload_msg
                    }
                else:
                    self.logger.warning(f"[VOLCENGINE_SESSION] 会话开始响应中未找到dialog_id")
                
                self.logger.info(f"[VOLCENGINE_SESSION] 会话开始完整响应: {payload_msg}")
                
        except Exception as e:
            self.logger.error(f"[VOLCENGINE_SESSION] 处理会话开始事件时出错: {e}")
    
    def _handle_chat_session_end(self, context: VolcEngineRealtimeContext):
        """处理聊天session结束（可在会话结束事件中调用）"""
        if hasattr(context, 'chat_session_active') and context.chat_session_active:
            self._flush_chat_buffer(context)
            self.logger.info(f"[VOLCENGINE_CHAT] 聊天session结束")
    
    async def _some_method_that_needs_dialog_id(self, context: VolcEngineRealtimeContext):
        """需要使用dialog_id的方法示例"""
        if self.current_dialog_id:
            # 使用dialog_id进行API调用或状态管理
            self.logger.info(f"使用dialog_id进行操作: {self.current_dialog_id}")
        else:
            self.logger.warning("当前没有可用的dialog_id")
    
    async def _handle_audio_data(self, context: VolcEngineRealtimeContext, audio_data: bytes):
        """处理接收到的音频数据"""
        try:
            if not audio_data:
                self.logger.warning(f"[VOLCENGINE_AUDIO] 收到空音频数据")
                return
            
            self.logger.info(f"[VOLCENGINE_AUDIO] 收到音频数据，长度: {len(audio_data)} 字节")
            
            # 将音频数据放入输出队列
            # 注意：火山引擎返回的是16位PCM数据，但我们需要转换为32位浮点格式
            context.audio_output_queue.put({
                'type': 'tts_audio',
                'data': audio_data,
                'timestamp': time.time(),
                'format': 'pcm_s16le',  # 明确标记为16位有符号小端格式
                'sample_rate': 24000,
                'channels': 1,
                'bits_per_sample': 16,
                'target_format': 'float32'  # 目标格式为32位浮点
            })
            
            self.logger.debug(f"[VOLCENGINE_AUDIO] 音频数据已放入输出队列")
            
        except Exception as e:
            self.logger.error(f"[VOLCENGINE_AUDIO] 处理音频数据时出错: {e}")
    
    async def _handle_server_responses(self, context: VolcEngineRealtimeContext):
        """处理服务器响应的异步任务"""
        self.logger.info(f"[VOLCENGINE_RESPONSE] 开始服务器响应处理循环: {context.session_id}")
        response_count = 0
        
        while not context.stop_event.is_set():
            try:
                # 接收服务器响应
                response = await context.client.receive_server_response()
                response_count += 1
                self.logger.info(f"[VOLCENGINE_RESPONSE] 收到服务器响应#{response_count}: {response}")
                
                # 处理不同类型的响应
                await self._process_server_response(context, response)
                
            except Exception as e:
                self.logger.error(f"[VOLCENGINE_RESPONSE] 接收服务器响应时出错: {e}")
                import traceback
                self.logger.error(f"[VOLCENGINE_RESPONSE] 响应处理错误堆栈: {traceback.format_exc()}")
                # 如果是连接错误，退出循环
                if "connection" in str(e).lower() or "websocket" in str(e).lower():
                    break
                # 其他错误继续尝试
                await asyncio.sleep(0.1)
                continue
        
        self.logger.info(f"[VOLCENGINE_RESPONSE] 服务器响应处理循环结束: {context.session_id}")
    
    async def _process_server_response(self, context: VolcEngineRealtimeContext, response: Dict[str, Any]):
        """处理具体的服务器响应"""
        try:
            # 根据响应类型处理
            if "text" in response:
                # 处理文本响应
                await self._on_message_received(response)
            elif "audio" in response:
                # 处理音频响应
                audio_data = response.get("audio", b"")
                if audio_data:
                    await self._on_audio_received(audio_data)
            else:
                # 其他类型响应的日志记录
                self.logger.info(f"[VOLCENGINE_RESPONSE] 收到其他类型响应: {response}")
        except Exception as e:
            self.logger.error(f"[VOLCENGINE_RESPONSE] 处理服务器响应时出错: {e}")
    
    def _process_received_responses(self, context: VolcEngineRealtimeContext, output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        """处理接收到的响应数据并输出到相应的数据类型"""
        try:
            # 处理文本输出队列中的ASR结果
            text_results = []
            while not context.text_output_queue.empty():
                try:
                    text_data = context.text_output_queue.get_nowait()
                    text_results.append(text_data)
                    self.logger.info(f"[VOLCENGINE_OUTPUT] 获取到ASR文本: {text_data}")
                except queue.Empty:
                    break
            
            # 输出ASR识别的文本
            if text_results and ChatDataType.AVATAR_TEXT in output_definitions:
                # 只输出最新的文本结果，避免重复累积
                for result in text_results:
                    text = result.get('text', '')
                    if text:
                        self.logger.info(f"[VOLCENGINE_OUTPUT] 输出ASR识别文本: '{text}'")
                        
                        # 创建文本输出数据
                        from chat_engine.data_models.runtime_data.data_bundle import DataBundle
                        data_bundle = DataBundle(definition=output_definitions[ChatDataType.AVATAR_TEXT].definition)
                        data_bundle.set_main_data(text)
                        data_bundle.add_meta('confidence', result.get('confidence', 0.0))
                        data_bundle.add_meta('source', 'volcengine_asr')
                        
                        text_data = ChatData(
                            source="volcengine_handler",
                            type=ChatDataType.AVATAR_TEXT,
                            timestamp=time.time(),
                            data=data_bundle
                        )
                        
                        # 输出文本数据
                        context.submit_data(text_data)
                        self.logger.info(f"[VOLCENGINE_OUTPUT] ASR文本已输出: {text}")
            
            # 处理音频输出队列中的TTS结果
            audio_results = []
            while not context.audio_output_queue.empty():
                try:
                    audio_data = context.audio_output_queue.get_nowait()
                    audio_results.append(audio_data)
                    self.logger.info(f"[VOLCENGINE_OUTPUT] 获取到TTS音频数据: {len(audio_data.get('data', b''))} 字节")
                except queue.Empty:
                    break
            
            # 输出TTS合成的音频
            if audio_results and ChatDataType.AVATAR_AUDIO in output_definitions:
                # 合并所有音频数据
                combined_audio = b''.join([result.get('data', b'') for result in audio_results if result.get('data')])
                if combined_audio:
                    self.logger.info(f"[VOLCENGINE_OUTPUT] 输出TTS合成音频: {len(combined_audio)} 字节")
                    
                    
                    # 创建音频输出数据
                    from chat_engine.data_models.runtime_data.data_bundle import DataBundle
                    # 火山引擎TTS返回的是32位float32 PCM数据，直接转换为numpy数组
                    audio_array = np.frombuffer(combined_audio, dtype=np.float32)
                    # 数据已经是32位浮点格式，无需额外转换，直接使用
                    # 注意：火山引擎返回的float32数据已经在[-1, 1]范围内
                    audio_array = audio_array[np.newaxis, ...]  # 添加声道维度，变为[1, samples]
                    data_bundle = DataBundle(definition=output_definitions[ChatDataType.AVATAR_AUDIO].definition)
                    data_bundle.set_main_data(audio_array)
                    data_bundle.add_meta('source', 'volcengine_tts')
                    data_bundle.add_meta('format', 'pcm')
                    data_bundle.add_meta('sample_rate', 24000)
                    data_bundle.add_meta('channels', 1)
                    data_bundle.add_meta('bits_per_sample', 32)
                    data_bundle.add_meta('data_type', 'float32')
                    
                    audio_data = ChatData(
                        source="volcengine_handler",
                        type=ChatDataType.AVATAR_AUDIO,
                        timestamp=time.time(),
                        data=data_bundle
                    )
                    
                    # 输出音频数据
                    context.submit_data(audio_data)
                    self.logger.info(f"[VOLCENGINE_OUTPUT] TTS音频已输出: {len(combined_audio)} 字节")
            
            # 记录处理结果
            self.logger.info(f"[VOLCENGINE_OUTPUT] 响应处理完成 - ASR文本: {len(text_results)}条, TTS音频: {len(audio_results)}条")
            
        except Exception as e:
            self.logger.error(f"[VOLCENGINE_OUTPUT] 处理响应数据时出错: {e}")
            import traceback
            self.logger.error(f"[VOLCENGINE_OUTPUT] 错误堆栈: {traceback.format_exc()}")
    
    async def _on_error_received(self, error: Exception):
        """处理错误回调"""
        self.logger.error(f"火山引擎客户端错误: {error}")
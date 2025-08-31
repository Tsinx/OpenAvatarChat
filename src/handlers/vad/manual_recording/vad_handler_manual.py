import math
import os
from abc import ABC
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, cast

import numpy as np
from loguru import logger

from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDetail, HandlerDataInfo
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_engine_config_data import HandlerBaseConfigModel, ChatEngineConfigModel
from chat_engine.data_models.runtime_data.data_bundle import DataBundleDefinition, DataBundleEntry, DataBundle
from engine_utils.general_slicer import SliceContext, slice_data


@dataclass
class ManualRecordingConfigModel(HandlerBaseConfigModel):
    """手动录音配置模型"""
    enabled: bool = True
    min_recording_duration_ms: int = 500  # 最小录音时长（毫秒）
    max_recording_duration_ms: int = 30000  # 最大录音时长（毫秒）
    buffer_size_ms: int = 100  # 音频缓冲区大小（毫秒）


class ManualRecordingContext(HandlerContext):
    """手动录音上下文"""
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: Optional[ManualRecordingConfigModel] = None
        self.recording_start_time: Optional[float] = None
        self.is_recording: bool = False
        self.audio_buffer: list = []
        self.slice_context: Optional[SliceContext] = None
        self.speech_id: int = 0


class HandlerManualRecording(HandlerBase, ABC):
    """手动录音处理器"""
    
    def __init__(self):
        super().__init__()
    
    def get_handler_info(self):
        return HandlerBaseInfo(
            config_model=ManualRecordingConfigModel
        )
    
    def load(self, engine_config: ChatEngineConfigModel, handler_config=None):
        """加载处理器"""
        logger.info("Manual recording handler loaded")
    
    def create_context(self, session_context: SessionContext, handler_config=None) -> HandlerContext:
        """创建上下文"""
        context = ManualRecordingContext(session_context.session_info.session_id)
        context.shared_states = session_context.shared_states
        
        if isinstance(handler_config, ManualRecordingConfigModel):
            context.config = handler_config
        else:
            context.config = ManualRecordingConfigModel()
        
        # 创建音频切片上下文
        buffer_size_samples = int(context.config.buffer_size_ms * 16)  # 16kHz采样率
        context.slice_context = SliceContext.create_numpy_slice_context(
            slice_size=buffer_size_samples,
            slice_axis=0,
        )
        
        return context
    
    def start_context(self, session_context, handler_context):
        """启动上下文"""
        pass
    
    def get_handler_detail(self, session_context: SessionContext,
                          context: HandlerContext) -> HandlerDetail:
        """获取处理器详情"""
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry("human_audio", 1, 16000))
        
        inputs = {
            ChatDataType.MIC_AUDIO: HandlerDataInfo(
                type=ChatDataType.MIC_AUDIO
            )
        }
        outputs = {
            ChatDataType.HUMAN_AUDIO: HandlerDataInfo(
                type=ChatDataType.HUMAN_AUDIO,
                definition=definition
            )
        }
        return HandlerDetail(
            inputs=inputs,
            outputs=outputs,
        )
    
    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        """处理音频数据"""
        context = cast(ManualRecordingContext, context)
        output_definition = output_definitions.get(ChatDataType.HUMAN_AUDIO).definition
        
        if not context.config.enabled:
            return
        
        if inputs.type != ChatDataType.MIC_AUDIO:
            return
        
        audio = inputs.data.get_main_data()
        if audio is None:
            return
        
        audio_entry = inputs.data.get_main_definition_entry()
        sample_rate = audio_entry.sample_rate
        audio = audio.squeeze()
        
        timestamp = None
        if inputs.is_timestamp_valid():
            timestamp = inputs.timestamp
        
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / 32767
        
        # 检查是否有录音控制信号
        recording_signal = inputs.data.get_meta('recording_signal')
        if recording_signal == 'start':
            # 开始录音
            context.is_recording = True
            context.recording_start_time = timestamp[0] if timestamp else 0
            context.audio_buffer = []
            context.speech_id += 1
            logger.info(f"Manual recording started, speech_id: {context.speech_id}")
        
        elif recording_signal == 'stop':
            # 停止录音
            if context.is_recording:
                context.is_recording = False
                recording_duration = 0
                if context.recording_start_time and timestamp:
                    recording_duration = (timestamp[0] - context.recording_start_time) / sample_rate * 1000
                
                # 检查录音时长
                if recording_duration >= context.config.min_recording_duration_ms:
                    # 合并录音数据
                    if context.audio_buffer:
                        combined_audio = np.concatenate(context.audio_buffer)
                        
                        # 创建输出数据
                        output = DataBundle(output_definition)
                        output.set_main_data(np.expand_dims(combined_audio, axis=0))
                        output.add_meta("human_speech_end", True)
                        output.add_meta("speech_id", f"manual-speech-{context.session_id}-{context.speech_id}")
                        output.add_meta("recording_duration_ms", recording_duration)
                        
                        output_chat_data = ChatData(
                            type=ChatDataType.HUMAN_AUDIO,
                            data=output
                        )
                        if timestamp:
                            output_chat_data.timestamp = timestamp
                        
                        logger.info(f"Manual recording completed, duration: {recording_duration:.1f}ms")
                        yield output_chat_data
                    else:
                        logger.warning("Recording stopped but no audio data collected")
                else:
                    logger.info(f"Recording too short ({recording_duration:.1f}ms), ignored")
                
                # 清空缓冲区
                context.audio_buffer = []
        
        # 如果正在录音，收集音频数据
        if context.is_recording:
            context.audio_buffer.append(audio)
            
            # 检查是否超过最大录音时长
            if context.recording_start_time and timestamp:
                recording_duration = (timestamp[0] - context.recording_start_time) / sample_rate * 1000
                if recording_duration > context.config.max_recording_duration_ms:
                    logger.warning(f"Recording exceeded max duration ({recording_duration:.1f}ms), auto-stopping")
                    context.is_recording = False
                    context.audio_buffer = []
    
    def destroy_context(self, context: HandlerContext):
        """销毁上下文"""
        pass

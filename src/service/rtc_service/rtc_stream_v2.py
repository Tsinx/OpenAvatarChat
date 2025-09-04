import asyncio
import json
import uuid
import weakref
from typing import Optional, Dict

import numpy as np
# noinspection PyPackageRequirements
from fastrtc import AsyncAudioVideoStreamHandler, AudioEmitType, VideoEmitType
from loguru import logger

from chat_engine.common.client_handler_base import ClientHandlerDelegate, ClientSessionDelegate
from chat_engine.common.engine_channel_type import EngineChannelType
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_signal import ChatSignal
from chat_engine.data_models.chat_signal_type import ChatSignalType, ChatSignalSourceType
from engine_utils.interval_counter import IntervalCounter
from aiortc.codecs import vpx 
vpx.DEFAULT_BITRATE = 5000000
vpx.MIN_BITRATE = 1000000
vpx.MAX_BITRATE = 10000000


class RtcStreamV2(AsyncAudioVideoStreamHandler):
    """支持手动录音控制的RTC流处理器"""
    
    def __init__(self,
                 session_id: Optional[str],
                 expected_layout="mono",
                 input_sample_rate=16000,
                 output_sample_rate=24000,
                 output_frame_size=480,
                 fps=30,
                 stream_start_delay = 0.5,
                 ):
        super().__init__(
            expected_layout=expected_layout,
            input_sample_rate=input_sample_rate,
            output_sample_rate=output_sample_rate,
            output_frame_size=output_frame_size,
            fps=fps
        )
        self.client_handler_delegate: Optional[ClientHandlerDelegate] = None
        self.client_session_delegate: Optional[ClientSessionDelegate] = None

        self.weak_factory: Optional[weakref.ReferenceType[RtcStreamV2]] = None

        self.session_id = session_id
        self.stream_start_delay = stream_start_delay

        self.chat_channel = None
        self.first_audio_emitted = False

        self.quit = asyncio.Event()
        self.last_frame_time = 0

        self.emit_counter = IntervalCounter("emit counter")

        self.start_time = None
        self.timestamp_base = self.input_sample_rate

        self.streams: Dict[str, RtcStreamV2] = {}
        
        # 手动录音控制状态
        self.is_recording = False
        self.recording_start_time = None

    # copy is used as create_instance in fastrtc
    def copy(self, **kwargs) -> AsyncAudioVideoStreamHandler:
        try:
            if self.client_handler_delegate is None:
                raise Exception("ClientHandlerDelegate is not set.")
            session_id = kwargs.get("webrtc_id", None)
            if session_id is None:
                session_id = uuid.uuid4().hex
            new_stream = RtcStreamV2(
                session_id,
                expected_layout=self.expected_layout,
                input_sample_rate=self.input_sample_rate,
                output_sample_rate=self.output_sample_rate,
                output_frame_size=self.output_frame_size,
                fps=self.fps,
                stream_start_delay=self.stream_start_delay,
            )
            new_stream.weak_factory = weakref.ref(self)
            new_session_delegate = self.client_handler_delegate.start_session(
                session_id=session_id,
                timestamp_base=self.input_sample_rate,
            )
            new_stream.client_session_delegate = new_session_delegate
            if session_id in self.streams:
                msg = f"Stream {session_id} already exists."
                raise RuntimeError(msg)
            self.streams[session_id] = new_stream
            return new_stream
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to create stream: {e}")
            raise

    async def emit(self) -> AudioEmitType:
        try:
            if not self.first_audio_emitted:
                self.client_session_delegate.clear_data()
                self.first_audio_emitted = True

            while not self.quit.is_set():
                chat_data = await self.client_session_delegate.get_data(EngineChannelType.AUDIO)
                if chat_data is None or chat_data.data is None:
                    continue
                audio_array = chat_data.data.get_main_data()
                if audio_array is None:
                    continue
                sample_num = audio_array.shape[-1]
                self.emit_counter.add_property("emit_audio", sample_num / self.output_sample_rate)
                return self.output_sample_rate, audio_array
        except Exception as e:
            logger.opt(exception=e).error(f"Error in emit: ")
            raise

    async def video_emit(self) -> VideoEmitType:
        try:
            if not self.first_audio_emitted:
                await asyncio.sleep(0.1)
            self.emit_counter.add_property("emit_video")
            while not self.quit.is_set():
                video_frame_data: ChatData = await self.client_session_delegate.get_data(EngineChannelType.VIDEO)
                if video_frame_data is None or video_frame_data.data is None:
                    continue
                frame_data = video_frame_data.data.get_main_data().squeeze()
                if frame_data is None:
                    continue
                return frame_data
        except Exception as e:
            logger.opt(exception=e).error(f"Error in video_emit: ")
            raise

    async def receive(self, frame: tuple[int, np.ndarray]):
        """接收音频数据，支持手动录音控制"""
        if self.client_session_delegate is None:
            return
        timestamp = self.client_session_delegate.get_timestamp()
        if timestamp[0] / timestamp[1] < self.stream_start_delay:
            return
        _, array = frame
        
        # 检查是否有录音控制信号
        recording_signal = getattr(frame, 'recording_signal', None)
        
        # 创建数据包并添加录音控制元数据
        data_bundle = self.client_session_delegate.input_data_definitions[EngineChannelType.AUDIO]
        if recording_signal:
            data_bundle.add_meta('recording_signal', recording_signal)
        
        self.client_session_delegate.put_data(
            EngineChannelType.AUDIO,
            array,
            timestamp,
            self.input_sample_rate,
        )

    async def video_receive(self, frame):
        if self.client_session_delegate is None:
            return
        timestamp = self.client_session_delegate.get_timestamp()
        if timestamp[0] / timestamp[1] < self.stream_start_delay:
            return
        self.client_session_delegate.put_data(
            EngineChannelType.VIDEO,
            frame,
            timestamp,
            self.fps,
        )

    def set_channel(self, channel):
        super().set_channel(channel)
        self.chat_channel = channel
        
        async def process_chat_history():
            role = None
            chat_id = None
            while not self.quit.is_set():
                chat_data = await self.client_session_delegate.get_data(EngineChannelType.TEXT)
                if chat_data is None or chat_data.data is None:
                    continue
                logger.debug(f"Got chat data {str(chat_data)}")
                current_role = 'human' if chat_data.type == ChatDataType.HUMAN_TEXT else 'avatar'
                chat_id = uuid.uuid4().hex if current_role != role else chat_id
                role = current_role
                self.chat_channel.send(json.dumps({'type': 'chat', 'message': chat_data.data.get_main_data(), 
                                                    'id': chat_id, 'role': current_role}))  
        asyncio.create_task(process_chat_history())
            
        @channel.on("message")
        def _(message):
            logger.info(f"Received message Custom: {message}")
            try:
                message = json.loads(message)
            except Exception as e:
                logger.info(e)
                message = {}

            if self.client_session_delegate is None:
                return
            timestamp = self.client_session_delegate.get_timestamp()
            if timestamp[0] / timestamp[1] < self.stream_start_delay:
                return
            logger.info(f'on_chat_datachannel: {message}')

            if message['type'] == 'stop_chat':
                self.client_session_delegate.emit_signal(
                    ChatSignal(
                        signal_type=ChatSignalType.STOP_CHAT,
                        source_type=ChatSignalSourceType.CLIENT,
                        source_id=self.session_id,
                        target_id=self.session_id,
                        data={}
                    )
                )
            elif message['type'] == 'recording_control':
                # 处理录音控制信号
                logger.info(f"🎤 [Backend] 接收到录音控制信号: {message}")
                control_action = message.get('action')
                duration = message.get('duration', 'N/A')
                
                if control_action == 'start':
                    logger.info(f"🎤 [Backend] 开始手动录音 - session_id: {self.session_id}")
                    self.is_recording = True
                    self.recording_start_time = timestamp[0]
                    logger.info(f"🎤 [Backend] 录音状态已设置: is_recording={self.is_recording}, start_time={self.recording_start_time}")
                elif control_action == 'stop':
                    logger.info(f"🎤 [Backend] 停止手动录音 - session_id: {self.session_id}, 前端报告时长: {duration}ms")
                    if self.is_recording:
                        actual_duration = timestamp[0] - self.recording_start_time if self.recording_start_time else 0
                        logger.info(f"🎤 [Backend] 后端计算的实际录音时长: {actual_duration}ms")
                    else:
                        logger.warning(f"🎤 [Backend] 警告: 收到停止信号但当前未在录音状态")
                    
                    self.is_recording = False
                    self.recording_start_time = None
                    logger.info(f"🎤 [Backend] 录音状态已重置: is_recording={self.is_recording}, start_time={self.recording_start_time}")
                else:
                    logger.warning(f"🎤 [Backend] 未知的录音控制动作: {control_action}")

    def close(self):
        logger.info(f"🔌 [Stream] 开始关闭RTC流 - session_id: {self.session_id}")
        
        # 设置退出信号
        logger.info(f"🔌 [Stream] 设置退出信号 - session_id: {self.session_id}")
        self.quit.set()
        logger.info(f"🔌 [Stream] 退出信号已设置 - session_id: {self.session_id}")
        
        # 从流字典中移除当前会话
        if self.session_id in self.streams:
            logger.info(f"🔌 [Stream] 从流字典中移除会话 - session_id: {self.session_id}")
            logger.info(f"🔌 [Stream] 移除前流字典大小: {len(self.streams)}")
            del self.streams[self.session_id]
            logger.info(f"🔌 [Stream] 移除后流字典大小: {len(self.streams)}")
        else:
            logger.warning(f"⚠️ [Stream] 警告: 会话不在流字典中 - session_id: {self.session_id}")
        
        # 调用父类的close方法
        logger.info(f"🔌 [Stream] 调用父类close方法 - session_id: {self.session_id}")
        super().close()
        logger.info(f"✅ [Stream] RTC流关闭完成 - session_id: {self.session_id}")

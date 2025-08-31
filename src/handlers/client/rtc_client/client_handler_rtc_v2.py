import asyncio
from pathlib import Path
from typing import Dict, Optional, cast, Union, Tuple
from uuid import uuid4

from loguru import logger

from engine_utils.directory_info import DirectoryInfo
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import gradio
import numpy as np
from fastapi import FastAPI
# noinspection PyPackageRequirements
from fastrtc import Stream

from pydantic import BaseModel, Field

from chat_engine.common.client_handler_base import ClientHandlerBase, ClientSessionDelegate
from chat_engine.common.engine_channel_type import EngineChannelType
from chat_engine.common.handler_base import HandlerDataInfo, HandlerDetail, HandlerBaseInfo
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_engine_config_data import HandlerBaseConfigModel, ChatEngineConfigModel
from chat_engine.data_models.chat_signal import ChatSignal
from chat_engine.data_models.runtime_data.data_bundle import DataBundleDefinition, DataBundleEntry, VariableSize, \
    DataBundle
from service.rtc_service.rtc_provider import RTCProvider
from service.rtc_service.rtc_stream_v2 import RtcStreamV2


class RtcClientSessionDelegate(ClientSessionDelegate):
    def __init__(self):
        self.timestamp_generator = None
        self.data_submitter = None
        self.shared_states = None
        self.output_queues = {
            EngineChannelType.AUDIO: asyncio.Queue(),
            EngineChannelType.VIDEO: asyncio.Queue(),
            EngineChannelType.TEXT: asyncio.Queue(),
        }
        self.input_data_definitions: Dict[EngineChannelType, DataBundleDefinition] = {}
        self.modality_mapping = {
            EngineChannelType.AUDIO: ChatDataType.MIC_AUDIO,
            EngineChannelType.VIDEO: ChatDataType.CAMERA_VIDEO,
            EngineChannelType.TEXT: ChatDataType.HUMAN_TEXT,
        }

    async def get_data(self, modality: EngineChannelType, timeout: Optional[float] = 0.1) -> Optional[ChatData]:
        data_queue = self.output_queues.get(modality)
        if data_queue is None:
            return None
        if timeout is not None and timeout > 0:
            try:
                data = await asyncio.wait_for(data_queue.get(), timeout)
            except asyncio.TimeoutError:
                return None
        else:
            data = await data_queue.get()
        return data

    def put_data(self, modality: EngineChannelType, data: Union[np.ndarray, str],
                 timestamp: Optional[Tuple[int, int]] = None, samplerate: Optional[int] = None, loopback: bool = False):
        if timestamp is None:
            timestamp = self.get_timestamp()
        if self.data_submitter is None:
            return
        definition = self.input_data_definitions.get(modality)
        chat_data_type = self.modality_mapping.get(modality)
        if chat_data_type is None or definition is None:
            return
        data_bundle = DataBundle(definition)
        if modality == EngineChannelType.AUDIO:
            data_bundle.set_main_data(data.squeeze()[np.newaxis, ...])
        elif modality == EngineChannelType.VIDEO:
            data_bundle.set_main_data(data[np.newaxis, ...])
        elif modality == EngineChannelType.TEXT:
            data_bundle.add_meta('human_text_end', True)
            data_bundle.add_meta('speech_id', str(uuid4()))
            data_bundle.set_main_data(data)
        else:
            return
        chat_data = ChatData(
            source="client",
            type=chat_data_type,
            data=data_bundle,
            timestamp=timestamp,
        )
        self.data_submitter.submit(chat_data)
        if loopback:
            self.output_queues[modality].put_nowait(chat_data)

    def get_timestamp(self):
        return self.timestamp_generator()

    def emit_signal(self, signal: ChatSignal):
        pass

    def clear_data(self):
        for data_queue in self.output_queues.values():
            while not data_queue.empty():
                data_queue.get_nowait()


class ClientRtcConfigModel(HandlerBaseConfigModel, BaseModel):
    connection_ttl: int = Field(default=900)
    turn_config: Optional[Dict] = Field(default=None)


class ClientRtcContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: Optional[ClientRtcConfigModel] = None
        self.client_session_delegate: Optional[RtcClientSessionDelegate] = None


class ClientHandlerRtcV2(ClientHandlerBase):
    """支持手动录音控制的RTC客户端处理器"""
    
    def __init__(self):
        super().__init__()
        self.engine_config = None
        self.handler_config = None
        self.rtc_streamer_factory: Optional[RtcStreamV2] = None

        self.output_bundle_definitions: Dict[EngineChannelType, DataBundleDefinition] = {}

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=ClientRtcConfigModel,
            client_session_delegate_class=RtcClientSessionDelegate,
        )

    def prepare_rtc_definitions(self):
        self.rtc_streamer_factory = RtcStreamV2(
            session_id=None,
            expected_layout="mono",
            input_sample_rate=16000,
            output_sample_rate=24000,
            output_frame_size=480,
            fps=30,
            stream_start_delay=0.5,
        )
        self.rtc_streamer_factory.client_handler_delegate = self.handler_delegate

        audio_output_definition = DataBundleDefinition()
        audio_output_definition.add_entry(DataBundleEntry.create_audio_entry(
            "mic_audio",
            1,
            16000,
        ))
        audio_output_definition.lockdown()
        self.output_bundle_definitions[EngineChannelType.AUDIO] = audio_output_definition

        video_output_definition = DataBundleDefinition()
        video_output_definition.add_entry(DataBundleEntry.create_framed_entry(
            "camera_video",
            [VariableSize(), VariableSize(), VariableSize(), 3],
            0,
            30
        ))
        video_output_definition.lockdown()
        self.output_bundle_definitions[EngineChannelType.VIDEO] = video_output_definition

        text_output_definition = DataBundleDefinition()
        text_output_definition.add_entry(DataBundleEntry.create_text_entry(
            "human_text",
        ))
        text_output_definition.lockdown()
        self.output_bundle_definitions[EngineChannelType.TEXT] = text_output_definition

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[HandlerBaseConfigModel] = None):
        self.engine_config = engine_config
        self.handler_config = cast(ClientRtcConfigModel, handler_config)
        self.prepare_rtc_definitions()

    def setup_rtc_ui(self, ui, parent_block, fastapi: FastAPI, avatar_config):
        turn_entity = RTCProvider().prepare_rtc_configuration(self.handler_config.turn_config)
        if turn_entity is None:
            turn_entity = RTCProvider().prepare_rtc_configuration(self.engine_config.turn_config)

        webrtc = Stream(
            modality="audio-video",
            mode="send-receive",
            time_limit=self.handler_config.connection_ttl,
            rtc_configuration=turn_entity.rtc_configuration if turn_entity is not None else None,
            handler=self.rtc_streamer_factory,
            concurrency_limit=self.handler_config.concurrent_limit,
        )
        webrtc.mount(fastapi)

        @fastapi.get('/openavatarchat/initconfig')
        async def init_config():
            config = {
                "avatar_config": avatar_config,
                "rtc_configuration": turn_entity.rtc_configuration if turn_entity is not None else None,
            }
            return JSONResponse(status_code=200, content=config)

        frontend_path = Path(DirectoryInfo.get_src_dir() + '/handlers/client/rtc_client/frontend/dist')
        if frontend_path.exists():
            fastapi.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="static")
        else:
            logger.warning(f"Frontend path {frontend_path} does not exist")

        @fastapi.get("/")
        async def root():
            return RedirectResponse(url="/index.html")

        return webrtc

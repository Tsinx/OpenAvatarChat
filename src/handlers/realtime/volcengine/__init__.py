# -*- coding: utf-8 -*-
"""
火山引擎实时对话处理器模块

该模块提供火山引擎实时对话的完整实现，包括：
1. 实时音频处理和格式转换
2. WebSocket连接和协议处理
3. ASR/LLM/TTS集成
4. 会话管理和状态跟踪
5. 音频录制和播放
6. 协议解析和处理

音频格式支持：
- ASR输入：16000Hz, 单声道, 16bit, 小端序
- TTS输出：24000Hz, 单声道, 32bit float32, 小端序
"""

from .realtime_handler_volcengine import HandlerVolcEngineRealtime
from .volcengine_client import VolcEngineRealtimeClient
from .protocol_handler import ProtocolHandler
from .protocol_parser import ProtocolParser
from .audio_processor import AudioProcessor, AudioConfig, AudioDeviceManager, DialogSession
from .audio_recorder import AudioRecorder, AudioPlayer
from .session_manager import SessionManager

__all__ = [
    'HandlerVolcEngineRealtime',
    'VolcEngineRealtimeClient',
    'ProtocolHandler', 
    'ProtocolParser',
    'AudioProcessor',
    'AudioConfig',
    'AudioDeviceManager',
    'DialogSession',
    'AudioRecorder',
    'AudioPlayer',
    'SessionManager'
]

__version__ = '1.0.0'
__author__ = 'OpenAvatarChat Team'
__description__ = '火山引擎实时对话处理器'
__audio_formats__ = {
    'asr_input': {'sample_rate': 16000, 'channels': 1, 'bit_depth': 16, 'format': 'int16', 'endian': 'little'},
    'tts_output': {'sample_rate': 24000, 'channels': 1, 'bit_depth': 32, 'format': 'float32', 'endian': 'little'}
}

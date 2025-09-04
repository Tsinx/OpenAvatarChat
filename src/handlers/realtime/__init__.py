# -*- coding: utf-8 -*-
"""
实时语音处理器模块

该模块提供各种实时语音处理器的实现，包括：
- 火山引擎实时对话处理器
- 其他实时语音处理器（待扩展）
"""

from .volcengine.realtime_handler_volcengine import HandlerVolcEngineRealtime

__all__ = [
    'HandlerVolcEngineRealtime'
]

__version__ = '1.0.0'
__description__ = '实时语音处理器模块'
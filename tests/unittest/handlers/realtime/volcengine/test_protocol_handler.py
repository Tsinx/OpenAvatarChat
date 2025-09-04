# -*- coding: utf-8 -*-
"""
火山引擎协议处理器单元测试

测试VolcEngineProtocolHandler类的核心功能:
1. 音频消息编码
2. 完整请求编码
3. 消息解码
4. 协议头处理
"""

import sys
import os
import unittest
import struct
import json
import numpy as np
import importlib.util
from unittest.mock import Mock, patch, MagicMock

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..')))

# 直接导入协议处理器文件，避免复杂的模块依赖
protocol_handler_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'src', 'handlers', 'realtime', 'volcengine', 'protocol_handler.py')
spec = importlib.util.spec_from_file_location("protocol_handler", protocol_handler_path)
protocol_handler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol_handler)

# 从模块中导入需要的类
VolcEngineProtocolHandler = protocol_handler.VolcEngineProtocolHandler
MessageType = protocol_handler.MessageType
AudioFormat = protocol_handler.AudioFormat


class TestVolcEngineProtocolHandler(unittest.TestCase):
    """VolcEngineProtocolHandler测试类"""
    
    def setUp(self):
        """测试前的设置"""
        self.handler = VolcEngineProtocolHandler(sample_rate=16000, channels=1, bit_depth=16)
        
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.handler.sample_rate, 16000)
        self.assertEqual(self.handler.channels, 1)
        self.assertEqual(self.handler.bit_depth, 16)
        self.assertEqual(self.handler.sequence_id, 0)
        self.assertEqual(self.handler.audio_format, AudioFormat.PCM_16K_16BIT_MONO)
        
    def test_audio_format_mapping(self):
        """测试音频格式映射"""
        # 测试8kHz
        handler_8k = VolcEngineProtocolHandler(sample_rate=8000)
        self.assertEqual(handler_8k.audio_format, AudioFormat.PCM_8K_16BIT_MONO)
        
        # 测试16kHz
        handler_16k = VolcEngineProtocolHandler(sample_rate=16000)
        self.assertEqual(handler_16k.audio_format, AudioFormat.PCM_16K_16BIT_MONO)
        
        # 测试24kHz
        handler_24k = VolcEngineProtocolHandler(sample_rate=24000)
        self.assertEqual(handler_24k.audio_format, AudioFormat.PCM_24K_16BIT_MONO)
        
        # 测试不支持的采样率，应该默认为16kHz
        handler_unknown = VolcEngineProtocolHandler(sample_rate=44100)
        self.assertEqual(handler_unknown.audio_format, AudioFormat.PCM_16K_16BIT_MONO)
        
    def test_encode_audio_message(self):
        """测试音频消息编码"""
        # 创建测试音频数据
        audio_data = np.array([1000, 2000, 3000, 4000], dtype=np.int16)
        
        # 编码音频消息
        encoded = self.handler.encode_audio_message(audio_data, is_last=False, request_id="test-123")
        
        # 验证消息结构
        self.assertGreater(len(encoded), 16)  # 至少包含16字节消息头
        
        # 验证消息头
        header = encoded[:16]
        magic, msg_type, flags, seq_id, payload_len, req_hash = struct.unpack('>4sBBHII', header)
        
        self.assertEqual(magic, b'VOLC')
        self.assertEqual(msg_type, MessageType.AUDIO_ONLY_CLIENT_REQUEST.value)
        self.assertEqual(flags, 0x00)  # is_last=False
        self.assertEqual(payload_len, len(audio_data.tobytes()))
        
        # 验证载荷数据
        payload = encoded[16:]
        self.assertEqual(payload, audio_data.tobytes())
        
    def test_encode_audio_message_last(self):
        """测试最后一个音频消息编码"""
        audio_data = np.array([1000, 2000], dtype=np.int16)
        
        encoded = self.handler.encode_audio_message(audio_data, is_last=True, request_id="test-456")
        
        # 验证is_last标志
        header = encoded[:16]
        _, _, flags, _, _, _ = struct.unpack('>4sBBHII', header)
        self.assertEqual(flags, 0x01)  # is_last=True
        
    def test_encode_audio_message_float32_conversion(self):
        """测试float32音频数据转换"""
        # 创建float32格式的音频数据
        audio_data = np.array([0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        
        encoded = self.handler.encode_audio_message(audio_data)
        
        # 验证转换后的数据
        payload = encoded[16:]
        converted_data = np.frombuffer(payload, dtype=np.int16)
        
        # 验证转换结果
        expected = np.array([16383, -16384, 32767, -32767], dtype=np.int16)
        np.testing.assert_array_almost_equal(converted_data, expected, decimal=0)
        
    def test_encode_full_request(self):
        """测试完整请求编码"""
        audio_data = np.array([1000, 2000, 3000], dtype=np.int16)
        text_data = "测试文本"
        
        encoded = self.handler.encode_full_request(audio_data, text_data, is_last=True, request_id="full-test")
        
        # 验证消息头
        header = encoded[:16]
        magic, msg_type, flags, seq_id, payload_len, req_hash = struct.unpack('>4sBBHII', header)
        
        self.assertEqual(magic, b'VOLC')
        self.assertEqual(msg_type, MessageType.FULL_CLIENT_REQUEST.value)
        self.assertEqual(flags, 0x01)  # is_last=True
        
        # 验证JSON载荷
        payload = encoded[16:]
        payload_data = json.loads(payload.decode('utf-8'))
        
        self.assertEqual(payload_data['audio_format'], AudioFormat.PCM_16K_16BIT_MONO.value)
        self.assertEqual(payload_data['sample_rate'], 16000)
        self.assertEqual(payload_data['channels'], 1)
        self.assertEqual(payload_data['text_data'], text_data)
        self.assertEqual(payload_data['is_last'], True)
        self.assertIn('audio_data', payload_data)
        
    def test_sequence_id_increment(self):
        """测试序列号递增"""
        audio_data = np.array([1000], dtype=np.int16)
        
        # 发送多个消息，验证序列号递增
        encoded1 = self.handler.encode_audio_message(audio_data)
        encoded2 = self.handler.encode_audio_message(audio_data)
        encoded3 = self.handler.encode_audio_message(audio_data)
        
        # 提取序列号
        _, _, _, seq1, _, _ = struct.unpack('>4sBBHII', encoded1[:16])
        _, _, _, seq2, _, _ = struct.unpack('>4sBBHII', encoded2[:16])
        _, _, _, seq3, _, _ = struct.unpack('>4sBBHII', encoded3[:16])
        
        self.assertEqual(seq1, 1)
        self.assertEqual(seq2, 2)
        self.assertEqual(seq3, 3)
        
    def test_sequence_id_wraparound(self):
        """测试序列号循环"""
        # 设置序列号接近最大值
        self.handler.sequence_id = 65535
        
        audio_data = np.array([1000], dtype=np.int16)
        encoded = self.handler.encode_audio_message(audio_data)
        
        # 验证序列号回到0
        _, _, _, seq_id, _, _ = struct.unpack('>4sBBHII', encoded[:16])
        self.assertEqual(seq_id, 0)
        self.assertEqual(self.handler.sequence_id, 0)
        
    def test_message_type_values(self):
        """测试消息类型枚举值"""
        self.assertEqual(MessageType.AUDIO_ONLY_SERVER_ACK.value, 0x11)
        self.assertEqual(MessageType.AUDIO_ONLY_CLIENT_REQUEST.value, 0x12)
        self.assertEqual(MessageType.FULL_CLIENT_REQUEST.value, 0x13)
        self.assertEqual(MessageType.FULL_SERVER_RESPONSE.value, 0x14)
        self.assertEqual(MessageType.ERROR_RESPONSE.value, 0x15)
        
    def test_audio_format_values(self):
        """测试音频格式枚举值"""
        self.assertEqual(AudioFormat.PCM_16K_16BIT_MONO.value, 1)
        self.assertEqual(AudioFormat.PCM_24K_16BIT_MONO.value, 2)
        self.assertEqual(AudioFormat.PCM_8K_16BIT_MONO.value, 3)
        
    def test_empty_audio_data(self):
        """测试空音频数据"""
        audio_data = np.array([], dtype=np.int16)
        
        encoded = self.handler.encode_audio_message(audio_data)
        
        # 验证消息头
        header = encoded[:16]
        _, _, _, _, payload_len, _ = struct.unpack('>4sBBHII', header)
        
        self.assertEqual(payload_len, 0)
        self.assertEqual(len(encoded), 16)  # 只有消息头
        
    def test_large_audio_data(self):
        """测试大音频数据"""
        # 创建较大的音频数据 (1秒16kHz音频)
        audio_data = np.random.randint(-32768, 32767, size=16000, dtype=np.int16)
        
        encoded = self.handler.encode_audio_message(audio_data)
        
        # 验证消息结构
        self.assertEqual(len(encoded), 16 + len(audio_data.tobytes()))
        
        # 验证载荷长度
        header = encoded[:16]
        _, _, _, _, payload_len, _ = struct.unpack('>4sBBHII', header)
        self.assertEqual(payload_len, len(audio_data.tobytes()))


if __name__ == '__main__':
    unittest.main()
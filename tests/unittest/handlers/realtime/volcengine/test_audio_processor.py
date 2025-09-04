# -*- coding: utf-8 -*-
"""
火山引擎音频处理器单元测试

测试AudioProcessor类的核心功能:
1. 音频格式转换
2. 重采样处理
3. 缓冲区管理
4. 音频预处理
"""

import sys
import os
import unittest
import numpy as np
import io
import struct
import importlib.util
from unittest.mock import Mock, patch, MagicMock

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..')))

# 直接导入音频处理器文件，避免复杂的模块依赖
audio_processor_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'src', 'handlers', 'realtime', 'volcengine', 'audio_processor.py')
spec = importlib.util.spec_from_file_location("audio_processor", audio_processor_path)
audio_processor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audio_processor)

# 从模块中导入需要的类
AudioProcessor = audio_processor.AudioProcessor


class TestAudioProcessor(unittest.TestCase):
    """AudioProcessor测试类"""
    
    def setUp(self):
        """测试前的设置"""
        self.processor = AudioProcessor(
            target_sample_rate=16000,
            target_channels=1,
            chunk_size=1024,
            overlap_size=256
        )
        
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.processor.target_sample_rate, 16000)
        self.assertEqual(self.processor.target_channels, 1)
        self.assertEqual(self.processor.chunk_size, 1024)
        self.assertEqual(self.processor.overlap_size, 256)
        self.assertEqual(len(self.processor.audio_buffer), 0)
        self.assertEqual(self.processor.processed_samples, 0)
        
    def test_process_audio_int16_conversion(self):
        """测试int16音频数据转换"""
        # 创建int16格式的测试数据
        audio_data = np.array([1000, -2000, 3000, -4000], dtype=np.int16)
        sample_rate = 16000
        
        processed = self.processor.process_audio(audio_data, sample_rate)
        
        # 验证转换为float32
        self.assertEqual(processed.dtype, np.float32)
        
        # 验证数值转换正确性
        expected = audio_data.astype(np.float32) / 32768.0
        np.testing.assert_array_almost_equal(processed, expected, decimal=5)
        
    def test_process_audio_int32_conversion(self):
        """测试int32音频数据转换"""
        audio_data = np.array([100000, -200000, 300000, -400000], dtype=np.int32)
        sample_rate = 16000
        
        processed = self.processor.process_audio(audio_data, sample_rate)
        
        # 验证转换为float32
        self.assertEqual(processed.dtype, np.float32)
        
        # 验证数值转换正确性
        expected = audio_data.astype(np.float32) / 2147483648.0
        np.testing.assert_array_almost_equal(processed, expected, decimal=5)
        
    def test_process_audio_multichannel_to_mono(self):
        """测试多声道转单声道"""
        # 创建立体声测试数据 (2通道)
        left_channel = np.array([1000, 2000, 3000], dtype=np.int16)
        right_channel = np.array([500, 1000, 1500], dtype=np.int16)
        stereo_data = np.column_stack([left_channel, right_channel])
        
        processed = self.processor.process_audio(stereo_data, 16000)
        
        # 验证转换为单声道
        self.assertEqual(len(processed.shape), 1)
        
        # 验证平均值计算正确
        expected_mono = (left_channel + right_channel) / 2.0
        expected_float = expected_mono.astype(np.float32) / 32768.0
        np.testing.assert_array_almost_equal(processed, expected_float, decimal=5)
        
    def test_simple_resample(self):
        """测试简单重采样功能"""
        # 创建8kHz的测试数据
        audio_data = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        
        # 使用简单重采样从8kHz到16kHz
        resampled = self.processor._simple_resample(audio_data, 8000, 16000)
        
        # 验证输出长度大约是输入的2倍
        self.assertGreater(len(resampled), len(audio_data))
        self.assertLessEqual(len(resampled), len(audio_data) * 2 + 1)
        
    def test_add_to_buffer_single_chunk(self):
        """测试添加单个音频块到缓冲区"""
        # 创建一个完整的chunk大小的音频数据
        audio_chunk = np.random.rand(1024).astype(np.float32)
        
        processed_chunks = self.processor.add_to_buffer(audio_chunk)
        
        # 应该返回一个处理过的chunk
        self.assertEqual(len(processed_chunks), 1)
        self.assertEqual(processed_chunks[0].dtype, np.int16)
        self.assertEqual(len(processed_chunks[0]), 1024)
        
    def test_add_to_buffer_multiple_chunks(self):
        """测试添加多个音频块到缓冲区"""
        # 创建大于chunk_size的音频数据
        large_audio = np.random.rand(2500).astype(np.float32)
        
        processed_chunks = self.processor.add_to_buffer(large_audio)
        
        # 应该返回多个处理过的chunks
        self.assertGreater(len(processed_chunks), 1)
        for chunk in processed_chunks:
            self.assertEqual(chunk.dtype, np.int16)
            self.assertEqual(len(chunk), 1024)
            
    def test_add_to_buffer_small_chunk(self):
        """测试添加小音频块到缓冲区"""
        # 创建小于chunk_size的音频数据
        small_audio = np.random.rand(500).astype(np.float32)
        
        processed_chunks = self.processor.add_to_buffer(small_audio)
        
        # 不应该返回任何chunk，因为缓冲区还没满
        self.assertEqual(len(processed_chunks), 0)
        self.assertEqual(len(self.processor.audio_buffer), 500)
        
    def test_flush_buffer_with_data(self):
        """测试刷新包含数据的缓冲区"""
        # 添加一些数据到缓冲区
        small_audio = np.random.rand(500).astype(np.float32)
        self.processor.add_to_buffer(small_audio)
        
        # 刷新缓冲区
        final_chunk = self.processor.flush_buffer()
        
        # 应该返回最终的chunk
        self.assertIsNotNone(final_chunk)
        self.assertEqual(final_chunk.dtype, np.int16)
        self.assertEqual(len(final_chunk), 1024)  # 应该被填充到chunk_size
        
        # 缓冲区应该被清空
        self.assertEqual(len(self.processor.audio_buffer), 0)
        
    def test_flush_buffer_empty(self):
        """测试刷新空缓冲区"""
        final_chunk = self.processor.flush_buffer()
        
        # 应该返回None
        self.assertIsNone(final_chunk)
        
    def test_reset_buffer(self):
        """测试重置缓冲区"""
        # 添加一些数据到缓冲区
        audio_data = np.random.rand(500).astype(np.float32)
        self.processor.add_to_buffer(audio_data)
        self.processor.processed_samples = 1000
        
        # 重置缓冲区
        self.processor.reset_buffer()
        
        # 验证缓冲区被清空
        self.assertEqual(len(self.processor.audio_buffer), 0)
        self.assertEqual(self.processor.processed_samples, 0)
        
    def test_normalize_audio(self):
        """测试音频归一化"""
        # 创建测试音频数据
        audio_data = np.array([0.1, 0.5, -0.3, 0.8, -0.9], dtype=np.float32)
        
        normalized = self.processor._normalize_audio(audio_data)
        
        # 验证归一化后的数据类型
        self.assertEqual(normalized.dtype, np.float32)
        self.assertEqual(len(normalized), len(audio_data))
        
        # 验证没有超出范围的值
        self.assertTrue(np.all(np.abs(normalized) <= 1.0))
        
    def test_apply_window(self):
        """测试窗函数应用"""
        # 创建测试音频块
        audio_chunk = np.ones(1024, dtype=np.float32)
        
        windowed = self.processor._apply_window(audio_chunk)
        
        # 验证窗函数效果
        self.assertEqual(windowed.dtype, np.float32)
        self.assertEqual(len(windowed), len(audio_chunk))
        
        # 窗函数应该在边缘处有衰减
        self.assertLess(windowed[0], windowed[512])  # 边缘值应该小于中心值
        self.assertLess(windowed[-1], windowed[512])
        
    def test_float32_to_int16_conversion(self):
        """测试float32到int16转换"""
        # 创建float32测试数据
        float_data = np.array([0.5, -0.5, 1.0, -1.0, 0.0], dtype=np.float32)
        
        int16_data = self.processor._float32_to_int16(float_data)
        
        # 验证转换结果
        self.assertEqual(int16_data.dtype, np.int16)
        self.assertEqual(len(int16_data), len(float_data))
        
        # 验证数值转换正确性
        expected = np.array([16383, -16384, 32767, -32767, 0], dtype=np.int16)
        np.testing.assert_array_equal(int16_data, expected)
        
    def test_float32_to_int16_clipping(self):
        """测试float32到int16转换的截断处理"""
        # 创建超出范围的float32数据
        float_data = np.array([1.5, -1.5, 2.0, -2.0], dtype=np.float32)
        
        int16_data = self.processor._float32_to_int16(float_data)
        
        # 验证截断到有效范围
        self.assertTrue(np.all(int16_data >= -32767))
        self.assertTrue(np.all(int16_data <= 32767))
        
    def test_parse_basic_wav_header(self):
        """测试基本WAV文件头解析"""
        # 创建简单的WAV文件头
        sample_rate = 16000
        channels = 1
        bit_depth = 16
        
        # WAV文件头结构
        wav_header = b'RIFF'
        wav_header += struct.pack('<I', 36)  # 文件大小-8
        wav_header += b'WAVE'
        wav_header += b'fmt '
        wav_header += struct.pack('<I', 16)  # fmt chunk大小
        wav_header += struct.pack('<H', 1)   # PCM格式
        wav_header += struct.pack('<H', channels)
        wav_header += struct.pack('<I', sample_rate)
        wav_header += struct.pack('<I', sample_rate * channels * bit_depth // 8)  # 字节率
        wav_header += struct.pack('<H', channels * bit_depth // 8)  # 块对齐
        wav_header += struct.pack('<H', bit_depth)
        wav_header += b'data'
        wav_header += struct.pack('<I', 8)  # 数据大小
        
        # 添加一些PCM数据
        pcm_data = struct.pack('<4h', 1000, -1000, 2000, -2000)
        wav_bytes = wav_header + pcm_data
        
        # 解析WAV数据
        audio_data, parsed_sample_rate = self.processor._parse_basic_wav(wav_bytes)
        
        # 验证解析结果
        self.assertEqual(parsed_sample_rate, sample_rate)
        self.assertEqual(audio_data.dtype, np.float32)
        self.assertEqual(len(audio_data), 4)
        
    def test_get_audio_info(self):
        """测试获取音频信息"""
        # 创建测试音频数据
        audio_data = np.random.rand(16000).astype(np.float32)  # 1秒的16kHz音频
        sample_rate = 16000
        
        info = self.processor.get_audio_info(audio_data, sample_rate)
        
        # 验证音频信息
        self.assertIn('duration', info)
        self.assertIn('sample_rate', info)
        self.assertIn('channels', info)
        self.assertIn('samples', info)
        self.assertIn('rms_level', info)
        self.assertIn('peak_level', info)
        
        self.assertAlmostEqual(info['duration'], 1.0, places=2)
        self.assertEqual(info['sample_rate'], sample_rate)
        self.assertEqual(info['samples'], len(audio_data))
        
    def test_buffer_overlap_processing(self):
        """测试缓冲区重叠处理"""
        # 创建足够大的音频数据来测试重叠
        audio_data = np.random.rand(2048).astype(np.float32)
        
        processed_chunks = self.processor.add_to_buffer(audio_data)
        
        # 验证处理了多个chunk
        self.assertGreater(len(processed_chunks), 1)
        
        # 验证缓冲区中还有剩余数据（由于重叠）
        self.assertGreater(len(self.processor.audio_buffer), 0)
        
    def test_empty_audio_processing(self):
        """测试空音频数据处理"""
        empty_audio = np.array([], dtype=np.float32)
        
        processed = self.processor.process_audio(empty_audio, 16000)
        
        # 验证空数据处理
        self.assertEqual(len(processed), 0)
        self.assertEqual(processed.dtype, np.float32)
        
    def test_different_sample_rates(self):
        """测试不同采样率的处理"""
        # 测试8kHz处理器
        processor_8k = AudioProcessor(target_sample_rate=8000)
        audio_data = np.random.rand(1000).astype(np.float32)
        
        processed = processor_8k.process_audio(audio_data, 8000)
        
        # 验证处理结果
        self.assertEqual(processed.dtype, np.float32)
        self.assertGreater(len(processed), 0)
        
    def test_large_audio_processing(self):
        """测试大音频数据处理"""
        # 创建大音频数据 (10秒的16kHz音频)
        large_audio = np.random.rand(160000).astype(np.float32)
        
        processed = self.processor.process_audio(large_audio, 16000)
        
        # 验证大数据处理
        self.assertEqual(processed.dtype, np.float32)
        self.assertEqual(len(processed), len(large_audio))


class TestAudioProcessorWithoutLibraries(unittest.TestCase):
    """测试没有音频库时的AudioProcessor功能"""
    
    def setUp(self):
        """测试前的设置"""
        # 模拟没有音频库的情况
        self.original_libs_available = audio_processor.AUDIO_LIBS_AVAILABLE
        audio_processor.AUDIO_LIBS_AVAILABLE = False
        
        self.processor = AudioProcessor()
        
    def tearDown(self):
        """测试后的清理"""
        # 恢复原始状态
        audio_processor.AUDIO_LIBS_AVAILABLE = self.original_libs_available
        
    def test_load_audio_file_without_libraries(self):
        """测试没有音频库时加载文件"""
        with self.assertRaises(RuntimeError):
            self.processor.load_audio_file("test.wav")
            
    def test_simple_resample_fallback(self):
        """测试简单重采样回退功能"""
        audio_data = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        
        # 测试从8kHz到16kHz的简单重采样
        processed = self.processor.process_audio(audio_data, 8000)
        
        # 验证处理结果
        self.assertEqual(processed.dtype, np.float32)
        self.assertGreater(len(processed), len(audio_data))


if __name__ == '__main__':
    unittest.main()
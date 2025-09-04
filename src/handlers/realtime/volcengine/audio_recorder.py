import pyaudio
import threading
import time
from typing import Optional, Callable
from loguru import logger


class AudioRecorder:
    """音频录制器，基于官方示例的AudioDeviceManager"""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1, 
                 chunk_size: int = 1024, format: int = pyaudio.paInt16):
        """
        初始化音频录制器
        
        Args:
            sample_rate: 采样率
            channels: 声道数
            chunk_size: 音频块大小
            format: 音频格式
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = format
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.recording_thread = None
        
        # 回调函数
        self.audio_callback: Optional[Callable[[bytes], None]] = None
        
        logger.info(f"AudioRecorder initialized: {sample_rate}Hz, {channels} channels, chunk_size={chunk_size}")
    
    def set_audio_callback(self, callback: Callable[[bytes], None]):
        """设置音频数据回调函数"""
        self.audio_callback = callback
    
    def start_recording(self, callback: Optional[Callable[[bytes], None]] = None):
        """
        开始录制音频
        
        Args:
            callback: 音频数据回调函数
        """
        if self.is_recording:
            logger.warning("Recording is already in progress")
            return
        
        if callback:
            self.audio_callback = callback
        
        if not self.audio_callback:
            logger.error("No audio callback set")
            return
        
        try:
            # 创建音频流
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            self.is_recording = True
            
            # 启动录制线程
            self.recording_thread = threading.Thread(target=self._recording_loop)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            logger.info("Audio recording started")
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.is_recording = False
            if self.stream:
                self.stream.close()
                self.stream = None
            raise
    
    def stop_recording(self):
        """停止录制音频"""
        if not self.is_recording:
            logger.warning("Recording is not in progress")
            return
        
        self.is_recording = False
        
        # 等待录制线程结束
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=1.0)
        
        # 关闭音频流
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error closing audio stream: {e}")
            finally:
                self.stream = None
        
        logger.info("Audio recording stopped")
    
    def _recording_loop(self):
        """录制循环"""
        logger.debug("Recording loop started")
        
        try:
            while self.is_recording and self.stream:
                try:
                    # 读取音频数据
                    audio_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # 调用回调函数
                    if self.audio_callback and audio_data:
                        self.audio_callback(audio_data)
                    
                except Exception as e:
                    if self.is_recording:  # 只有在录制状态下才报告错误
                        logger.error(f"Error reading audio data: {e}")
                    break
        
        except Exception as e:
            logger.error(f"Recording loop error: {e}")
        
        finally:
            logger.debug("Recording loop ended")
    
    def is_recording_active(self) -> bool:
        """检查是否正在录制"""
        return self.is_recording
    
    def get_audio_info(self) -> dict:
        """获取音频配置信息"""
        return {
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'chunk_size': self.chunk_size,
            'format': self.format,
            'is_recording': self.is_recording
        }
    
    def close(self):
        """关闭音频录制器"""
        self.stop_recording()
        
        try:
            self.audio.terminate()
        except Exception as e:
            logger.error(f"Error terminating PyAudio: {e}")
        
        logger.info("AudioRecorder closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 兼容性函数
def create_audio_recorder(sample_rate: int = 16000, channels: int = 1, 
                         chunk_size: int = 1024) -> AudioRecorder:
    """创建音频录制器的便捷函数"""
    return AudioRecorder(
        sample_rate=sample_rate,
        channels=channels,
        chunk_size=chunk_size
    )


class AudioPlayer:
    """音频播放器，基于官方示例的AudioDeviceManager"""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1, 
                 chunk_size: int = 1024, format: int = pyaudio.paInt16):
        """
        初始化音频播放器
        
        Args:
            sample_rate: 采样率
            channels: 声道数
            chunk_size: 音频块大小
            format: 音频格式
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = format
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_playing = False
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        
        logger.info(f"AudioPlayer initialized: {sample_rate}Hz, {channels} channels, chunk_size={chunk_size}")
    
    def start_playback(self):
        """开始播放音频"""
        if self.is_playing:
            logger.warning("Playback is already in progress")
            return
        
        try:
            def audio_callback(in_data, frame_count, time_info, status):
                with self.buffer_lock:
                    if self.audio_buffer:
                        data = self.audio_buffer.pop(0)
                        return (data, pyaudio.paContinue)
                    else:
                        # 返回静音数据
                        silence = b'\x00' * frame_count * self.channels * 2
                        return (silence, pyaudio.paContinue)
            
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=audio_callback
            )
            
            self.is_playing = True
            self.stream.start_stream()
            
            logger.info("Audio playback started")
            
        except Exception as e:
            logger.error(f"Failed to start playback: {e}")
            self.is_playing = False
            if self.stream:
                self.stream.close()
                self.stream = None
            raise
    
    def stop_playback(self):
        """停止播放音频"""
        if not self.is_playing:
            logger.warning("Playback is not in progress")
            return
        
        self.is_playing = False
        
        # 关闭音频流
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error closing audio stream: {e}")
            finally:
                self.stream = None
        
        # 清空缓冲区
        with self.buffer_lock:
            self.audio_buffer.clear()
        
        logger.info("Audio playback stopped")
    
    def add_audio_data(self, audio_data: bytes):
        """添加音频数据到播放缓冲区"""
        with self.buffer_lock:
            self.audio_buffer.append(audio_data)
    
    def is_playing_active(self) -> bool:
        """检查是否正在播放"""
        return self.is_playing
    
    def close(self):
        """关闭音频播放器"""
        self.stop_playback()
        
        try:
            self.audio.terminate()
        except Exception as e:
            logger.error(f"Error terminating PyAudio: {e}")
        
        logger.info("AudioPlayer closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
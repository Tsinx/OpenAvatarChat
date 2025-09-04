import pyaudio
import asyncio
import threading
import time
from typing import Dict, Any, Optional, Callable, Awaitable


class AudioConfig:
    def __init__(self, sample_rate: int = 16000, channels: int = 1, chunk_size: int = 1024, format: int = pyaudio.paInt16):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = format


class AudioDeviceManager:
    def __init__(self, audio_config: AudioConfig):
        self.audio_config = audio_config
        self.audio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self.is_recording = False
        self.is_playing = False
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()

    def start_recording(self, callback: Callable[[bytes], None]):
        if self.is_recording:
            return
        
        def audio_callback(in_data, frame_count, time_info, status):
            if self.is_recording:
                callback(in_data)
            return (None, pyaudio.paContinue)
        
        self.input_stream = self.audio.open(
            format=self.audio_config.format,
            channels=self.audio_config.channels,
            rate=self.audio_config.sample_rate,
            input=True,
            frames_per_buffer=self.audio_config.chunk_size,
            stream_callback=audio_callback
        )
        self.is_recording = True
        self.input_stream.start_stream()
        print("Recording started")

    def stop_recording(self):
        if self.input_stream and self.is_recording:
            self.is_recording = False
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.input_stream = None
            print("Recording stopped")

    def start_playback(self):
        if self.is_playing:
            return
        
        def audio_callback(in_data, frame_count, time_info, status):
            with self.buffer_lock:
                if self.audio_buffer:
                    data = self.audio_buffer.pop(0)
                    return (data, pyaudio.paContinue)
                else:
                    return (b'\x00' * frame_count * self.audio_config.channels * 2, pyaudio.paContinue)
        
        self.output_stream = self.audio.open(
            format=self.audio_config.format,
            channels=self.audio_config.channels,
            rate=self.audio_config.sample_rate,
            output=True,
            frames_per_buffer=self.audio_config.chunk_size,
            stream_callback=audio_callback
        )
        self.is_playing = True
        self.output_stream.start_stream()
        print("Playback started")

    def stop_playback(self):
        if self.output_stream and self.is_playing:
            self.is_playing = False
            self.output_stream.stop_stream()
            self.output_stream.close()
            self.output_stream = None
            with self.buffer_lock:
                self.audio_buffer.clear()
            print("Playback stopped")

    def add_audio_to_buffer(self, audio_data: bytes):
        with self.buffer_lock:
            self.audio_buffer.append(audio_data)

    def close(self):
        self.stop_recording()
        self.stop_playback()
        self.audio.terminate()


class DialogSession:
    def __init__(self, client, config: Dict[str, Any]):
        self.client = client
        self.config = config
        self.audio_config = AudioConfig(
            sample_rate=config.get('input_audio_config', {}).get('sample_rate', 16000),
            channels=config.get('input_audio_config', {}).get('channels', 1),
            chunk_size=config.get('input_audio_config', {}).get('chunk_size', 1024)
        )
        self.audio_manager = AudioDeviceManager(self.audio_config)
        self.is_running = False
        self.response_handler_task = None
        
        # 回调函数
        self.on_message_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
        self.on_audio_callback: Optional[Callable[[bytes], Awaitable[None]]] = None
        self.on_error_callback: Optional[Callable[[Exception], Awaitable[None]]] = None

    def set_callbacks(self, 
                     on_message: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
                     on_audio: Optional[Callable[[bytes], Awaitable[None]]] = None,
                     on_error: Optional[Callable[[Exception], Awaitable[None]]] = None):
        """设置回调函数"""
        self.on_message_callback = on_message
        self.on_audio_callback = on_audio
        self.on_error_callback = on_error

    async def start_session(self):
        """启动对话会话"""
        try:
            # 连接到服务器
            await self.client.connect()
            
            # 启动会话
            await self.client.start_session(self.config.get('start_session_req', {}))
            
            # 发送Hello消息
            await self.client.say_hello()
            
            self.is_running = True
            
            # 启动响应处理任务
            self.response_handler_task = asyncio.create_task(self._handle_server_responses())
            
            print("Dialog session started successfully")
            
        except Exception as e:
            print(f"Failed to start session: {e}")
            if self.on_error_callback:
                await self.on_error_callback(e)
            raise

    async def stop_session(self):
        """停止对话会话"""
        self.is_running = False
        
        # 停止音频设备
        self.audio_manager.close()
        
        # 取消响应处理任务
        if self.response_handler_task:
            self.response_handler_task.cancel()
            try:
                await self.response_handler_task
            except asyncio.CancelledError:
                pass
        
        # 断开客户端连接
        if hasattr(self.client, 'disconnect'):
            await self.client.disconnect()
        
        print("Dialog session stopped")

    async def start_microphone_input(self):
        """开始麦克风输入"""
        def on_audio_data(audio_data: bytes):
            # 在新的事件循环中发送音频数据
            asyncio.create_task(self.client.send_audio_data(audio_data))
        
        self.audio_manager.start_recording(on_audio_data)

    def stop_microphone_input(self):
        """停止麦克风输入"""
        self.audio_manager.stop_recording()

    async def send_text_message(self, text: str):
        """发送文本消息"""
        await self.client.send_text_query(text)

    async def process_audio_file(self, file_path: str):
        """处理音频文件"""
        try:
            import wave
            
            with wave.open(file_path, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                
                # 分块发送音频数据
                chunk_size = self.audio_config.chunk_size * 2  # 2 bytes per sample for 16-bit
                for i in range(0, len(frames), chunk_size):
                    chunk = frames[i:i + chunk_size]
                    await self.client.send_audio_data(chunk)
                    await asyncio.sleep(0.01)  # 小延迟模拟实时音频流
                    
        except Exception as e:
            print(f"Failed to process audio file: {e}")
            if self.on_error_callback:
                await self.on_error_callback(e)

    async def _handle_server_responses(self):
        """处理服务器响应"""
        while self.is_running:
            try:
                response = await self.client.receive_server_response()
                
                # 处理不同类型的响应
                if 'audio' in response:
                    audio_data = response['audio']
                    if isinstance(audio_data, str):
                        import base64
                        audio_data = base64.b64decode(audio_data)
                    
                    # 添加到播放缓冲区
                    self.audio_manager.add_audio_to_buffer(audio_data)
                    
                    # 如果还没开始播放，启动播放
                    if not self.audio_manager.is_playing:
                        self.audio_manager.start_playback()
                    
                    if self.on_audio_callback:
                        await self.on_audio_callback(audio_data)
                
                if 'text' in response or 'content' in response:
                    if self.on_message_callback:
                        await self.on_message_callback(response)
                
            except Exception as e:
                if self.is_running:  # 只有在会话运行时才报告错误
                    print(f"Error handling server response: {e}")
                    if self.on_error_callback:
                        await self.on_error_callback(e)
                break

# 兼容性类，保持与现有代码的接口一致
class AudioProcessor:
    """音频处理器兼容性类"""
    
    def __init__(self):
        pass
    
    def process_audio_bytes(self, audio_data: bytes, target_sample_rate: int = 16000, target_channels: int = 1):
        """处理音频字节数据（兼容性方法）"""
        # 简单返回原始数据，实际项目中可能需要重采样等处理
        import numpy as np
        return np.frombuffer(audio_data, dtype=np.int16)
    
    def load_audio_file(self, file_path: str):
        """加载音频文件（兼容性方法）"""
        try:
            import wave
            with wave.open(file_path, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                return frames
        except Exception as e:
            print(f"Failed to load audio file: {e}")
            return None
import asyncio
import time
from typing import Dict, Any, Optional, Callable, List
from loguru import logger
from audio_processor import DialogSession, AudioConfig, AudioDeviceManager
from volcengine_client import VolcEngineRealtimeClient
from protocol_handler import ProtocolHandler


class SessionManager:
    """会话管理器，基于官方示例的DialogSession实现"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化会话管理器
        
        Args:
            config: 会话配置
        """
        self.config = config
        self.sessions: Dict[str, DialogSession] = {}
        self.clients: Dict[str, VolcEngineRealtimeClient] = {}
        self.protocol_handlers: Dict[str, ProtocolHandler] = {}
        self.audio_managers: Dict[str, AudioDeviceManager] = {}
        
        # 会话状态跟踪
        self.session_states: Dict[str, str] = {}  # 'idle', 'connecting', 'active', 'closing'
        self.last_activity: Dict[str, float] = {}
        
        # 回调函数
        self.on_message_callback: Optional[Callable] = None
        self.on_audio_callback: Optional[Callable] = None
        self.on_error_callback: Optional[Callable] = None
        
        logger.info("SessionManager initialized")
    
    def set_callbacks(self, 
                     on_message: Optional[Callable] = None,
                     on_audio: Optional[Callable] = None,
                     on_error: Optional[Callable] = None):
        """
        设置回调函数
        
        Args:
            on_message: 消息回调函数
            on_audio: 音频回调函数
            on_error: 错误回调函数
        """
        self.on_message_callback = on_message
        self.on_audio_callback = on_audio
        self.on_error_callback = on_error
    
    async def create_session(self, session_id: str, 
                           session_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        创建新会话
        
        Args:
            session_id: 会话ID
            session_config: 会话特定配置
            
        Returns:
            bool: 创建是否成功
        """
        try:
            if session_id in self.sessions:
                logger.warning(f"Session {session_id} already exists")
                return False
            
            # 合并配置
            config = self.config.copy()
            if session_config:
                config.update(session_config)
            
            # 创建音频配置
            audio_config = AudioConfig(
                sample_rate=config.get('sample_rate', 16000),
                channels=config.get('channels', 1),
                chunk_size=config.get('chunk_size', 1024),
                format=config.get('audio_format', 'int16')
            )
            
            # 创建音频设备管理器
            audio_manager = AudioDeviceManager(audio_config)
            self.audio_managers[session_id] = audio_manager
            
            # 创建协议处理器
            protocol_handler = ProtocolHandler(
                use_compression=config.get('use_compression', True),
                use_protobuf=config.get('use_protobuf', False)
            )
            self.protocol_handlers[session_id] = protocol_handler
            
            # 创建客户端
            client = VolcEngineRealtimeClient(
                uri=config.get('websocket_url', ''),
                headers=config.get('headers', {})
            )
            self.clients[session_id] = client
            
            # 创建对话会话
            session = DialogSession(
                client=client,
                audio_manager=audio_manager,
                config=config
            )
            
            # 设置回调
            if self.on_message_callback:
                session.set_message_callback(self.on_message_callback)
            if self.on_audio_callback:
                session.set_audio_callback(self.on_audio_callback)
            if self.on_error_callback:
                session.set_error_callback(self.on_error_callback)
            
            self.sessions[session_id] = session
            self.session_states[session_id] = 'idle'
            self.last_activity[session_id] = time.time()
            
            logger.info(f"Session {session_id} created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create session {session_id}: {e}")
            await self._cleanup_session(session_id)
            return False
    
    async def start_session(self, session_id: str) -> bool:
        """
        启动会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            bool: 启动是否成功
        """
        try:
            if session_id not in self.sessions:
                logger.error(f"Session {session_id} not found")
                return False
            
            session = self.sessions[session_id]
            self.session_states[session_id] = 'connecting'
            
            # 启动会话
            success = await session.start()
            
            if success:
                self.session_states[session_id] = 'active'
                self.last_activity[session_id] = time.time()
                logger.info(f"Session {session_id} started successfully")
            else:
                self.session_states[session_id] = 'idle'
                logger.error(f"Failed to start session {session_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error starting session {session_id}: {e}")
            self.session_states[session_id] = 'idle'
            return False
    
    async def stop_session(self, session_id: str) -> bool:
        """
        停止会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            bool: 停止是否成功
        """
        try:
            if session_id not in self.sessions:
                logger.warning(f"Session {session_id} not found")
                return True
            
            session = self.sessions[session_id]
            self.session_states[session_id] = 'closing'
            
            # 停止会话
            await session.stop()
            
            # 清理资源
            await self._cleanup_session(session_id)
            
            logger.info(f"Session {session_id} stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping session {session_id}: {e}")
            return False
    
    async def send_audio(self, session_id: str, audio_data: bytes) -> bool:
        """
        发送音频数据
        
        Args:
            session_id: 会话ID
            audio_data: 音频数据
            
        Returns:
            bool: 发送是否成功
        """
        try:
            if session_id not in self.sessions:
                logger.error(f"Session {session_id} not found")
                return False
            
            if self.session_states.get(session_id) != 'active':
                logger.error(f"Session {session_id} is not active")
                return False
            
            session = self.sessions[session_id]
            success = await session.send_audio(audio_data)
            
            if success:
                self.last_activity[session_id] = time.time()
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending audio to session {session_id}: {e}")
            return False
    
    async def send_text(self, session_id: str, text: str, 
                       message_type: str = 'ChatTextQuery') -> bool:
        """
        发送文本消息
        
        Args:
            session_id: 会话ID
            text: 文本内容
            message_type: 消息类型
            
        Returns:
            bool: 发送是否成功
        """
        try:
            if session_id not in self.sessions:
                logger.error(f"Session {session_id} not found")
                return False
            
            if self.session_states.get(session_id) != 'active':
                logger.error(f"Session {session_id} is not active")
                return False
            
            session = self.sessions[session_id]
            
            if message_type == 'ChatTextQuery':
                success = await session.send_chat_text_query(text)
            elif message_type == 'ChatTTSText':
                success = await session.send_chat_tts_text(text)
            else:
                logger.error(f"Unsupported message type: {message_type}")
                return False
            
            if success:
                self.last_activity[session_id] = time.time()
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending text to session {session_id}: {e}")
            return False
    
    async def process_audio_file(self, session_id: str, file_path: str) -> bool:
        """
        处理音频文件
        
        Args:
            session_id: 会话ID
            file_path: 音频文件路径
            
        Returns:
            bool: 处理是否成功
        """
        try:
            if session_id not in self.sessions:
                logger.error(f"Session {session_id} not found")
                return False
            
            if self.session_states.get(session_id) != 'active':
                logger.error(f"Session {session_id} is not active")
                return False
            
            session = self.sessions[session_id]
            success = await session.process_audio_file(file_path)
            
            if success:
                self.last_activity[session_id] = time.time()
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing audio file for session {session_id}: {e}")
            return False
    
    async def start_microphone(self, session_id: str) -> bool:
        """
        启动麦克风录音
        
        Args:
            session_id: 会话ID
            
        Returns:
            bool: 启动是否成功
        """
        try:
            if session_id not in self.sessions:
                logger.error(f"Session {session_id} not found")
                return False
            
            session = self.sessions[session_id]
            success = await session.start_microphone()
            
            if success:
                self.last_activity[session_id] = time.time()
            
            return success
            
        except Exception as e:
            logger.error(f"Error starting microphone for session {session_id}: {e}")
            return False
    
    async def stop_microphone(self, session_id: str) -> bool:
        """
        停止麦克风录音
        
        Args:
            session_id: 会话ID
            
        Returns:
            bool: 停止是否成功
        """
        try:
            if session_id not in self.sessions:
                logger.error(f"Session {session_id} not found")
                return False
            
            session = self.sessions[session_id]
            success = await session.stop_microphone()
            
            return success
            
        except Exception as e:
            logger.error(f"Error stopping microphone for session {session_id}: {e}")
            return False
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话信息
        
        Args:
            session_id: 会话ID
            
        Returns:
            Dict[str, Any]: 会话信息，如果会话不存在返回None
        """
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        return {
            'session_id': session_id,
            'state': self.session_states.get(session_id, 'unknown'),
            'last_activity': self.last_activity.get(session_id, 0),
            'is_connected': session.is_connected() if hasattr(session, 'is_connected') else False,
            'config': self.config
        }
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        列出所有会话
        
        Returns:
            List[Dict[str, Any]]: 会话信息列表
        """
        sessions_info = []
        
        for session_id in self.sessions:
            info = self.get_session_info(session_id)
            if info:
                sessions_info.append(info)
        
        return sessions_info
    
    async def cleanup_inactive_sessions(self, timeout: float = 300.0):
        """
        清理不活跃的会话
        
        Args:
            timeout: 超时时间（秒）
        """
        current_time = time.time()
        inactive_sessions = []
        
        for session_id, last_activity in self.last_activity.items():
            if (current_time - last_activity) > timeout:
                inactive_sessions.append(session_id)
        
        for session_id in inactive_sessions:
            logger.info(f"Cleaning up inactive session: {session_id}")
            await self.stop_session(session_id)
    
    async def send_keepalive(self, session_id: str) -> bool:
        """
        发送保活消息
        
        Args:
            session_id: 会话ID
            
        Returns:
            bool: 发送是否成功
        """
        try:
            if session_id not in self.protocol_handlers:
                return False
            
            protocol_handler = self.protocol_handlers[session_id]
            
            # 检查是否需要发送保活消息
            current_time = time.time()
            last_activity = self.last_activity.get(session_id, 0)
            
            if protocol_handler.is_keepalive_needed(last_activity, current_time):
                # 创建保活消息
                keepalive_msg = protocol_handler.create_keepalive_message()
                
                # 通过会话发送
                if session_id in self.sessions:
                    session = self.sessions[session_id]
                    success = await session.send_raw_message(keepalive_msg)
                    
                    if success:
                        self.last_activity[session_id] = current_time
                        logger.debug(f"Keepalive sent for session {session_id}")
                    
                    return success
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending keepalive for session {session_id}: {e}")
            return False
    
    async def _cleanup_session(self, session_id: str):
        """
        清理会话资源
        
        Args:
            session_id: 会话ID
        """
        try:
            # 清理会话
            if session_id in self.sessions:
                del self.sessions[session_id]
            
            # 清理客户端
            if session_id in self.clients:
                client = self.clients[session_id]
                if hasattr(client, 'close'):
                    await client.close()
                del self.clients[session_id]
            
            # 清理协议处理器
            if session_id in self.protocol_handlers:
                del self.protocol_handlers[session_id]
            
            # 清理音频管理器
            if session_id in self.audio_managers:
                audio_manager = self.audio_managers[session_id]
                if hasattr(audio_manager, 'cleanup'):
                    audio_manager.cleanup()
                del self.audio_managers[session_id]
            
            # 清理状态
            if session_id in self.session_states:
                del self.session_states[session_id]
            
            if session_id in self.last_activity:
                del self.last_activity[session_id]
            
            logger.debug(f"Session {session_id} resources cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")
    
    async def shutdown(self):
        """
        关闭会话管理器，清理所有资源
        """
        logger.info("Shutting down SessionManager")
        
        # 停止所有会话
        session_ids = list(self.sessions.keys())
        for session_id in session_ids:
            await self.stop_session(session_id)
        
        # 清理所有资源
        self.sessions.clear()
        self.clients.clear()
        self.protocol_handlers.clear()
        self.audio_managers.clear()
        self.session_states.clear()
        self.last_activity.clear()
        
        logger.info("SessionManager shutdown complete")


# 兼容性函数
def create_session_manager(config: Dict[str, Any]) -> SessionManager:
    """创建会话管理器的便捷函数"""
    return SessionManager(config)


# 会话状态常量
SESSION_STATE_IDLE = 'idle'
SESSION_STATE_CONNECTING = 'connecting'
SESSION_STATE_ACTIVE = 'active'
SESSION_STATE_CLOSING = 'closing'
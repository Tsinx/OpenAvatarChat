import json
import gzip
from typing import Dict, Any, Optional, Union
from loguru import logger
from protocol_parser import (
    generate_header, parse_response, extract_audio_data, 
    extract_text_data, is_audio_response, is_text_response,
    MSG_TYPE_FULL_CLIENT_REQUEST, MSG_TYPE_AUDIO_ONLY_REQUEST,
    MSG_TYPE_FULL_SERVER_RESPONSE, MSG_TYPE_SERVER_ACK,
    MSG_TYPE_SERVER_ERROR_RESPONSE, SERIALIZATION_PROTOBUF,
    SERIALIZATION_JSON, COMPRESSION_GZIP, COMPRESSION_NONE
)


class ProtocolHandler:
    """协议处理器，基于官方示例的协议实现"""
    
    def __init__(self, use_compression: bool = True, use_protobuf: bool = False):
        """
        初始化协议处理器
        
        Args:
            use_compression: 是否使用gzip压缩
            use_protobuf: 是否使用protobuf序列化（当前仅支持JSON）
        """
        self.use_compression = use_compression
        self.use_protobuf = use_protobuf
        self.serialization = SERIALIZATION_PROTOBUF if use_protobuf else SERIALIZATION_JSON
        self.compression = COMPRESSION_GZIP if use_compression else COMPRESSION_NONE
        
        logger.info(f"ProtocolHandler initialized: compression={use_compression}, protobuf={use_protobuf}")
    
    def create_start_connection_message(self, config: Dict[str, Any]) -> bytes:
        """
        创建StartConnection消息
        
        Args:
            config: 连接配置
            
        Returns:
            bytes: 编码后的消息
        """
        message = {
            "type": "StartConnection",
            "data": config
        }
        return self._encode_message(message, MSG_TYPE_FULL_CLIENT_REQUEST)
    
    def create_start_session_message(self, session_config: Dict[str, Any]) -> bytes:
        """
        创建StartSession消息
        
        Args:
            session_config: 会话配置
            
        Returns:
            bytes: 编码后的消息
        """
        message = {
            "type": "StartSession",
            "data": session_config
        }
        return self._encode_message(message, MSG_TYPE_FULL_CLIENT_REQUEST)
    
    def create_hello_message(self) -> bytes:
        """
        创建Hello消息
        
        Returns:
            bytes: 编码后的消息
        """
        message = {
            "type": "Hello",
            "data": {}
        }
        return self._encode_message(message, MSG_TYPE_FULL_CLIENT_REQUEST)
    
    def create_chat_text_query_message(self, text: str, query_id: str = None) -> bytes:
        """
        创建ChatTextQuery消息
        
        Args:
            text: 查询文本
            query_id: 查询ID
            
        Returns:
            bytes: 编码后的消息
        """
        data = {"text": text}
        if query_id:
            data["query_id"] = query_id
            
        message = {
            "type": "ChatTextQuery",
            "data": data
        }
        return self._encode_message(message, MSG_TYPE_FULL_CLIENT_REQUEST)
    
    def create_chat_tts_text_message(self, text: str, query_id: str = None) -> bytes:
        """
        创建ChatTTSText消息
        
        Args:
            text: TTS文本
            query_id: 查询ID
            
        Returns:
            bytes: 编码后的消息
        """
        data = {"text": text}
        if query_id:
            data["query_id"] = query_id
            
        message = {
            "type": "ChatTTSText",
            "data": data
        }
        return self._encode_message(message, MSG_TYPE_FULL_CLIENT_REQUEST)
    
    def create_task_request_message(self, task_type: str, task_data: Dict[str, Any]) -> bytes:
        """
        创建TaskRequest消息
        
        Args:
            task_type: 任务类型
            task_data: 任务数据
            
        Returns:
            bytes: 编码后的消息
        """
        message = {
            "type": "TaskRequest",
            "data": {
                "task_type": task_type,
                **task_data
            }
        }
        return self._encode_message(message, MSG_TYPE_FULL_CLIENT_REQUEST)
    
    def create_audio_message(self, audio_data: bytes, sequence_id: int = 0) -> bytes:
        """
        创建音频消息
        
        Args:
            audio_data: 音频数据
            sequence_id: 序列ID
            
        Returns:
            bytes: 编码后的消息
        """
        # 音频消息使用特殊的头部格式
        header = generate_header(
            msg_type=MSG_TYPE_AUDIO_ONLY_REQUEST,
            msg_serialization=self.serialization,
            msg_compression=self.compression,
            reserved=sequence_id
        )
        
        # 音频数据可能需要压缩
        if self.use_compression:
            try:
                compressed_data = gzip.compress(audio_data)
                return header + compressed_data
            except Exception as e:
                logger.warning(f"Audio compression failed, using uncompressed: {e}")
                return header + audio_data
        else:
            return header + audio_data
    
    def create_finish_session_message(self) -> bytes:
        """
        创建FinishSession消息
        
        Returns:
            bytes: 编码后的消息
        """
        message = {
            "type": "FinishSession",
            "data": {}
        }
        return self._encode_message(message, MSG_TYPE_FULL_CLIENT_REQUEST)
    
    def create_finish_connection_message(self) -> bytes:
        """
        创建FinishConnection消息
        
        Returns:
            bytes: 编码后的消息
        """
        message = {
            "type": "FinishConnection",
            "data": {}
        }
        return self._encode_message(message, MSG_TYPE_FULL_CLIENT_REQUEST)
    
    def parse_server_message(self, data: bytes) -> Optional[Dict[str, Any]]:
        """
        解析服务器消息
        
        Args:
            data: 原始消息数据
            
        Returns:
            Dict[str, Any]: 解析后的消息，如果解析失败返回None
        """
        try:
            # 使用官方协议解析器
            response = parse_response(data)
            
            if not response:
                logger.warning("Failed to parse server response")
                return None
            
            # 提取消息类型和内容
            message_type = self._get_message_type_from_response(response)
            
            result = {
                "type": message_type,
                "raw_response": response
            }
            
            # 根据消息类型提取特定数据
            if is_audio_response(data):
                audio_data = extract_audio_data(data)
                if audio_data:
                    result["audio_data"] = audio_data
                    result["data_type"] = "audio"
            
            if is_text_response(data):
                text_data = extract_text_data(data)
                if text_data:
                    result["text_data"] = text_data
                    result["data_type"] = "text"
            
            # 尝试解析JSON内容
            if "payload" in response:
                try:
                    if isinstance(response["payload"], str):
                        json_data = json.loads(response["payload"])
                        result["json_data"] = json_data
                    elif isinstance(response["payload"], dict):
                        result["json_data"] = response["payload"]
                except (json.JSONDecodeError, TypeError) as e:
                    logger.debug(f"Failed to parse JSON payload: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing server message: {e}")
            return None
    
    def _encode_message(self, message: Dict[str, Any], msg_type: int) -> bytes:
        """
        编码消息
        
        Args:
            message: 消息内容
            msg_type: 消息类型
            
        Returns:
            bytes: 编码后的消息
        """
        try:
            # 序列化消息内容
            if self.use_protobuf:
                # TODO: 实现protobuf序列化
                raise NotImplementedError("Protobuf serialization not implemented")
            else:
                payload = json.dumps(message, ensure_ascii=False).encode('utf-8')
            
            # 压缩消息内容
            if self.use_compression:
                payload = gzip.compress(payload)
            
            # 生成消息头
            header = generate_header(
                msg_type=msg_type,
                msg_serialization=self.serialization,
                msg_compression=self.compression,
                reserved=0
            )
            
            return header + payload
            
        except Exception as e:
            logger.error(f"Error encoding message: {e}")
            raise
    
    def _get_message_type_from_response(self, response: Dict[str, Any]) -> str:
        """
        从响应中获取消息类型
        
        Args:
            response: 解析后的响应
            
        Returns:
            str: 消息类型
        """
        # 根据响应头部信息判断消息类型
        msg_type = response.get("msg_type", 0)
        
        if msg_type == MSG_TYPE_FULL_SERVER_RESPONSE:
            return "ServerResponse"
        elif msg_type == MSG_TYPE_SERVER_ACK:
            return "ServerAck"
        elif msg_type == MSG_TYPE_SERVER_ERROR_RESPONSE:
            return "ServerError"
        else:
            return "Unknown"
    
    def create_keepalive_message(self) -> bytes:
        """
        创建保活消息（320字节静默消息）
        
        Returns:
            bytes: 保活消息
        """
        # 创建320字节的静默音频数据
        silence_data = b'\x00' * 320
        return self.create_audio_message(silence_data)
    
    def is_keepalive_needed(self, last_activity_time: float, current_time: float, 
                           keepalive_interval: float = 30.0) -> bool:
        """
        检查是否需要发送保活消息
        
        Args:
            last_activity_time: 上次活动时间
            current_time: 当前时间
            keepalive_interval: 保活间隔（秒）
            
        Returns:
            bool: 是否需要发送保活消息
        """
        return (current_time - last_activity_time) >= keepalive_interval
    
    def get_protocol_info(self) -> Dict[str, Any]:
        """
        获取协议配置信息
        
        Returns:
            Dict[str, Any]: 协议配置信息
        """
        return {
            "use_compression": self.use_compression,
            "use_protobuf": self.use_protobuf,
            "serialization": self.serialization,
            "compression": self.compression
        }


# 兼容性函数
def create_protocol_handler(use_compression: bool = True, 
                          use_protobuf: bool = False) -> ProtocolHandler:
    """创建协议处理器的便捷函数"""
    return ProtocolHandler(
        use_compression=use_compression,
        use_protobuf=use_protobuf
    )


# 消息类型常量（向后兼容）
MESSAGE_TYPE_START_CONNECTION = "StartConnection"
MESSAGE_TYPE_START_SESSION = "StartSession"
MESSAGE_TYPE_HELLO = "Hello"
MESSAGE_TYPE_CHAT_TEXT_QUERY = "ChatTextQuery"
MESSAGE_TYPE_CHAT_TTS_TEXT = "ChatTTSText"
MESSAGE_TYPE_TASK_REQUEST = "TaskRequest"
MESSAGE_TYPE_AUDIO = "Audio"
MESSAGE_TYPE_FINISH_SESSION = "FinishSession"
MESSAGE_TYPE_FINISH_CONNECTION = "FinishConnection"
MESSAGE_TYPE_SERVER_RESPONSE = "ServerResponse"
MESSAGE_TYPE_SERVER_ACK = "ServerAck"
MESSAGE_TYPE_SERVER_ERROR = "ServerError"
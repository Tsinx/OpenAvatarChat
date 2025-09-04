import websockets
import gzip
import json

from typing import Dict, Any
from loguru import logger

from protocol_parser import generate_header, parse_response


class VolcEngineRealtimeClient:
    def __init__(self, config: Dict[str, Any], session_id: str, output_audio_format: str = "pcm") -> None:
        logger.info(f"[VOLCENGINE_CLIENT] 初始化客户端 - session_id: {session_id}, 音频格式: {output_audio_format}")
        self.config = config
        logger.info(f"config: {config}")
        self.logid = ""
        self.session_id = session_id
        self.output_audio_format = output_audio_format
        self.ws = None
        logger.debug(f"[VOLCENGINE_CLIENT] 客户端配置已设置 - base_url: {config.get('base_url', 'N/A')}")

    async def connect(self) -> None:
        """建立WebSocket连接"""
        logger.info(f"[VOLCENGINE_CLIENT] 开始连接到服务器: {self.config['ws_connect_config']['base_url']}")
        logger.debug(f"[VOLCENGINE_CLIENT] 连接参数 - URL: {self.config['ws_connect_config']['base_url']}, Headers: {self.config['ws_connect_config']['headers']}")
        self.ws = await websockets.connect(
            self.config['ws_connect_config']['base_url'],
            additional_headers=self.config['ws_connect_config']['headers'],
            ping_interval=None
        )
        # 在新版本websockets中，response_headers可能不可用，使用try-except处理
        try:
            self.logid = getattr(self.ws, 'response_headers', {}).get("X-Tt-Logid", "")
        except:
            self.logid = ""
        logger.info(f"[VOLCENGINE_CLIENT] WebSocket连接已建立 - logid: {self.logid}")

        # StartConnection request
        start_connection_request = bytearray(generate_header())
        start_connection_request.extend(int(1).to_bytes(4, 'big'))
        payload_bytes = str.encode("{}")
        payload_bytes = gzip.compress(payload_bytes)
        start_connection_request.extend((len(payload_bytes)).to_bytes(4, 'big'))
        start_connection_request.extend(payload_bytes)
        await self.ws.send(start_connection_request)
        response = await self.ws.recv()
        logger.info(f"[VOLCENGINE_CLIENT] StartConnection响应: {parse_response(response)}")
        start_session_req = self.config['start_session_req']
        if self.output_audio_format == "pcm_s16le":
            start_session_req["tts"]["audio_config"]["format"] = "pcm_s16le"
        request_params = start_session_req
        payload_bytes = str.encode(json.dumps(request_params))
        payload_bytes = gzip.compress(payload_bytes)
        start_session_request = bytearray(generate_header())
        start_session_request.extend(int(100).to_bytes(4, 'big'))
        start_session_request.extend((len(self.session_id)).to_bytes(4, 'big'))
        start_session_request.extend(str.encode(self.session_id))
        start_session_request.extend((len(payload_bytes)).to_bytes(4, 'big'))
        start_session_request.extend(payload_bytes)
        
        await self.ws.send(start_session_request)
        response = await self.ws.recv()
        logger.info(f"[VOLCENGINE_CLIENT] StartSession响应: {parse_response(response)}")

    async def start_connection(self) -> None:
        """启动连接（connect方法的别名）"""
        await self.connect()

    async def start_session(self, start_session_req: Dict[str, Any]) -> None:
        """启动会话"""
        logger.info(f"[VOLCENGINE_CLIENT] 启动会话 - session_id: {self.session_id}")
        # StartSession request

    async def say_hello(self) -> None:
        """发送Hello消息"""
        payload = {
            "content": "你好，我是豆包，有什么可以帮助你的？",
        }
        hello_request = bytearray(generate_header())
        hello_request.extend(int(300).to_bytes(4, 'big'))
        payload_bytes = str.encode(json.dumps(payload))
        payload_bytes = gzip.compress(payload_bytes)
        hello_request.extend((len(self.session_id)).to_bytes(4, 'big'))
        hello_request.extend(str.encode(self.session_id))
        hello_request.extend((len(payload_bytes)).to_bytes(4, 'big'))
        hello_request.extend(payload_bytes)
        await self.ws.send(hello_request)
        
    async def send_text_query(self, content: str) -> None:
        """发送Chat Text Query消息"""
        payload = {
            "content": content,
        }
        chat_text_query_request = bytearray(generate_header())
        chat_text_query_request.extend(int(501).to_bytes(4, 'big'))
        payload_bytes = str.encode(json.dumps(payload))
        payload_bytes = gzip.compress(payload_bytes)
        chat_text_query_request.extend((len(self.session_id)).to_bytes(4, 'big'))
        chat_text_query_request.extend(str.encode(self.session_id))
        chat_text_query_request.extend((len(payload_bytes)).to_bytes(4, 'big'))
        chat_text_query_request.extend(payload_bytes)
        await self.ws.send(chat_text_query_request)

    async def chat_tts_text(self, is_user_querying: bool, start: bool, end: bool, content: str) -> None:
        if is_user_querying:
            return
        """发送Chat TTS Text消息"""
        payload = {
            "start": start,
            "end": end,
            "content": content,
        }
        logger.debug(f"[VOLCENGINE_CLIENT] ChatTTSText请求载荷: {payload}")
        payload_bytes = str.encode(json.dumps(payload))
        payload_bytes = gzip.compress(payload_bytes)

        chat_tts_text_request = bytearray(generate_header())
        chat_tts_text_request.extend(int(500).to_bytes(4, 'big'))
        chat_tts_text_request.extend((len(self.session_id)).to_bytes(4, 'big'))
        chat_tts_text_request.extend(str.encode(self.session_id))
        chat_tts_text_request.extend((len(payload_bytes)).to_bytes(4, 'big'))
        chat_tts_text_request.extend(payload_bytes)
        await self.ws.send(chat_tts_text_request)

    async def send_audio_data(self, audio: bytes) -> None:
        """发送音频数据"""
        # logger.debug(f"[VOLCENGINE_CLIENT] 发送音频数据 - 长度: {len(audio)} 字节")
        from protocol_parser import CLIENT_AUDIO_ONLY_REQUEST, NO_SERIALIZATION
        task_request = bytearray(
            generate_header(message_type=CLIENT_AUDIO_ONLY_REQUEST,
                                     serial_method=NO_SERIALIZATION))
        task_request.extend(int(200).to_bytes(4, 'big'))
        task_request.extend((len(self.session_id)).to_bytes(4, 'big'))
        task_request.extend(str.encode(self.session_id))
        payload_bytes = gzip.compress(audio)
        task_request.extend((len(payload_bytes)).to_bytes(4, 'big'))  # payload size(4 bytes)
        task_request.extend(payload_bytes)
        await self.ws.send(task_request)

    async def receive_server_response(self) -> Dict[str, Any]:
        try:
            response = await self.ws.recv()
            data = parse_response(response)
            return data
        except Exception as e:
            raise Exception(f"Failed to receive message: {e}")

    async def finish_session(self):
        finish_session_request = bytearray(generate_header())
        finish_session_request.extend(int(102).to_bytes(4, 'big'))
        payload_bytes = str.encode("{}")
        payload_bytes = gzip.compress(payload_bytes)
        finish_session_request.extend((len(self.session_id)).to_bytes(4, 'big'))
        finish_session_request.extend(str.encode(self.session_id))
        finish_session_request.extend((len(payload_bytes)).to_bytes(4, 'big'))
        finish_session_request.extend(payload_bytes)
        await self.ws.send(finish_session_request)

    async def finish_connection(self):
        finish_connection_request = bytearray(generate_header())
        finish_connection_request.extend(int(2).to_bytes(4, 'big'))
        payload_bytes = str.encode("{}")
        payload_bytes = gzip.compress(payload_bytes)
        finish_connection_request.extend((len(payload_bytes)).to_bytes(4, 'big'))
        finish_connection_request.extend(payload_bytes)
        await self.ws.send(finish_connection_request)
        response = await self.ws.recv()
        logger.info(f"[VOLCENGINE_CLIENT] FinishConnection响应: {parse_response(response)}")

    async def close(self) -> None:
        """关闭WebSocket连接"""
        logger.info(f"[VOLCENGINE_CLIENT] 关闭WebSocket连接 - session_id: {self.session_id}")
        if self.ws:
            logger.info(f"[VOLCENGINE_CLIENT] 正在关闭WebSocket连接...")
            await self.ws.close()

    # 兼容性方法，保持与现有代码的接口一致
    async def disconnect(self) -> None:
        """断开连接（兼容性方法）"""
        await self.finish_session()
        await self.finish_connection()
        await self.close()
import gzip
import json
from typing import Dict, Any, Optional

PROTOCOL_VERSION = 0b0001
DEFAULT_HEADER_SIZE = 0b0001

PROTOCOL_VERSION_BITS = 4
HEADER_BITS = 4
MESSAGE_TYPE_BITS = 4
MESSAGE_TYPE_SPECIFIC_FLAGS_BITS = 4
MESSAGE_SERIALIZATION_BITS = 4
MESSAGE_COMPRESSION_BITS = 4
RESERVED_BITS = 8

# Message Type:
CLIENT_FULL_REQUEST = 0b0001
CLIENT_AUDIO_ONLY_REQUEST = 0b0010

SERVER_FULL_RESPONSE = 0b1001
SERVER_ACK = 0b1011
SERVER_ERROR_RESPONSE = 0b1111

# Message Type Constants for compatibility
MSG_TYPE_FULL_CLIENT_REQUEST = CLIENT_FULL_REQUEST
MSG_TYPE_AUDIO_ONLY_REQUEST = CLIENT_AUDIO_ONLY_REQUEST
MSG_TYPE_FULL_SERVER_RESPONSE = SERVER_FULL_RESPONSE
MSG_TYPE_SERVER_ACK = SERVER_ACK
MSG_TYPE_SERVER_ERROR_RESPONSE = SERVER_ERROR_RESPONSE

# Message Type Specific Flags
NO_SEQUENCE = 0b0000  # no check sequence
POS_SEQUENCE = 0b0001
NEG_SEQUENCE = 0b0010
NEG_SEQUENCE_1 = 0b0011

MSG_WITH_EVENT = 0b0100

# Message Serialization
NO_SERIALIZATION = 0b0000
JSON = 0b0001
THRIFT = 0b0011
CUSTOM_TYPE = 0b1111

# Serialization Constants for compatibility
SERIALIZATION_JSON = JSON
SERIALIZATION_PROTOBUF = THRIFT

# Message Compression
NO_COMPRESSION = 0b0000
GZIP = 0b0001
CUSTOM_COMPRESSION = 0b1111

# Compression Constants for compatibility
COMPRESSION_NONE = NO_COMPRESSION
COMPRESSION_GZIP = GZIP


def generate_header(
        version=PROTOCOL_VERSION,
        message_type=CLIENT_FULL_REQUEST,
        message_type_specific_flags=MSG_WITH_EVENT,
        serial_method=JSON,
        compression_type=GZIP,
        reserved_data=0x00,
        extension_header=bytes()
):
    """
    protocol_version(4 bits), header_size(4 bits),
    message_type(4 bits), message_type_specific_flags(4 bits)
    serialization_method(4 bits) message_compression(4 bits)
    reserved （8bits) 保留字段
    header_extensions 扩展头(大小等于 8 * 4 * (header_size - 1) )
    """
    header = bytearray()
    header_size = int(len(extension_header) / 4) + 1
    header.append((version << 4) | header_size)
    header.append((message_type << 4) | message_type_specific_flags)
    header.append((serial_method << 4) | compression_type)
    header.append(reserved_data)
    header.extend(extension_header)
    return header


def parse_response(res):
    """
    - header
        - (4bytes)header
        - (4bits)version(v1) + (4bits)header_size
        - (4bits)messageType + (4bits)messageTypeFlags
            -- 0001	CompleteClient | -- 0001 hasSequence
            -- 0010	audioonly      | -- 0010 isTailPacket
                                           | -- 0100 hasEvent
        - (4bits)payloadFormat + (4bits)compression
        - (8bits) reserve
    - payload
        - [optional 4 bytes] event
        - [optional] session ID
          -- (4 bytes)session ID len
          -- session ID data
        - (4 bytes)data len
        - data
    """
    if isinstance(res, str):
        return {}
    protocol_version = res[0] >> 4
    header_size = res[0] & 0x0f
    message_type = res[1] >> 4
    message_type_specific_flags = res[1] & 0x0f
    serialization_method = res[2] >> 4
    message_compression = res[2] & 0x0f
    reserved = res[3]
    header_extensions = res[4:header_size * 4]
    payload = res[header_size * 4:]
    result = {}
    payload_msg = None
    payload_size = 0
    start = 0
    if message_type == SERVER_FULL_RESPONSE or message_type == SERVER_ACK:
        result['message_type'] = 'SERVER_FULL_RESPONSE'
        if message_type == SERVER_ACK:
            result['message_type'] = 'SERVER_ACK'
        if message_type_specific_flags & NEG_SEQUENCE > 0:
            result['seq'] = int.from_bytes(payload[:4], "big", signed=False)
            start += 4
        if message_type_specific_flags & MSG_WITH_EVENT > 0:
            result['event'] = int.from_bytes(payload[:4], "big", signed=False)
            start += 4
        payload = payload[start:]
        session_id_size = int.from_bytes(payload[:4], "big", signed=True)
        session_id = payload[4:session_id_size+4]
        result['session_id'] = str(session_id)
        payload = payload[4 + session_id_size:]
        payload_size = int.from_bytes(payload[:4], "big", signed=False)
        payload_msg = payload[4:]
    elif message_type == SERVER_ERROR_RESPONSE:
        code = int.from_bytes(payload[:4], "big", signed=False)
        result['code'] = code
        payload_size = int.from_bytes(payload[4:8], "big", signed=False)
        payload_msg = payload[8:]
    if payload_msg is None:
        return result
    if message_compression == GZIP:
        payload_msg = gzip.decompress(payload_msg)
    if serialization_method == JSON:
        payload_msg = json.loads(str(payload_msg, "utf-8"))
    elif serialization_method != NO_SERIALIZATION:
        payload_msg = str(payload_msg, "utf-8")
    result['payload_msg'] = payload_msg
    result['payload_size'] = payload_size
    return result


# 兼容性函数，保持与现有代码的接口一致
def extract_audio_data(parsed_response: Dict[str, Any]) -> Optional[bytes]:
    """从解析的响应中提取音频数据"""
    payload = parsed_response.get('payload_msg', {})
    if isinstance(payload, dict) and 'audio' in payload:
        return payload['audio']
    return None


def extract_text_data(parsed_response: Dict[str, Any]) -> Optional[str]:
    """从解析的响应中提取文本数据"""
    payload = parsed_response.get('payload_msg', {})
    if isinstance(payload, dict):
        # 尝试多种可能的文本字段
        for field in ['text', 'content', 'message', 'result']:
            if field in payload:
                return payload[field]
    return None


def get_event_type(parsed_response: Dict[str, Any]) -> Optional[int]:
    """获取事件类型"""
    return parsed_response.get('event')


def is_audio_response(parsed_response: Dict[str, Any]) -> bool:
    """判断是否为音频响应"""
    payload = parsed_response.get('payload_msg', {})
    return isinstance(payload, dict) and 'audio' in payload


def is_text_response(parsed_response: Dict[str, Any]) -> bool:
    """判断是否为文本响应"""
    payload = parsed_response.get('payload_msg', {})
    if isinstance(payload, dict):
        return any(field in payload for field in ['text', 'content', 'message', 'result'])
    return False


def is_error_response(parsed_response: Dict[str, Any]) -> bool:
    """判断是否为错误响应"""
    return parsed_response.get('message_type') == 'SERVER_ERROR_RESPONSE'
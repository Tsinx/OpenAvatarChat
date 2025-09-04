#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试FinishSession消息编码问题
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from handlers.realtime.volcengine.protocol_handler import VolcEngineProtocolHandler
import json

def test_finish_session_encoding():
    """测试FinishSession消息编码"""
    print("=== 测试FinishSession消息编码 ===")
    
    # 创建协议处理器
    handler = VolcEngineProtocolHandler()
    
    # 测试数据
    request_data = {
        "session_id": "9509c2e1-b784-441f-9eac-e23205d4e593"
    }
    
    print(f"请求数据: {request_data}")
    
    try:
        # 编码消息
        message = handler.encode_control_message(
            "finish_session",
            request_data
        )
        
        print(f"编码后消息长度: {len(message)} 字节")
        print(f"消息头 (前16字节): {message[:16].hex()}")
        print(f"载荷部分: {message[16:]}")
        
        # 尝试解码载荷部分
        try:
            payload_str = message[16:].decode('utf-8')
            print(f"载荷字符串: {payload_str}")
            
            # 尝试解析JSON
            payload_json = json.loads(payload_str)
            print(f"载荷JSON: {payload_json}")
            
        except UnicodeDecodeError as e:
            print(f"载荷解码失败: {e}")
            print(f"载荷原始字节: {message[16:]}")
            
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            
        # 检查消息中是否有无效字节
        print("\n=== 字节分析 ===")
        for i, byte_val in enumerate(message):
            if byte_val > 127:  # 非ASCII字符
                print(f"位置 {i}: 0x{byte_val:02x} ({byte_val})")
                
        # 检查是否有无效的UTF-8序列
        print("\n=== UTF-8验证 ===")
        try:
            message.decode('utf-8')
            print("整个消息可以作为UTF-8解码")
        except UnicodeDecodeError as e:
            print(f"UTF-8解码失败: {e}")
            
    except Exception as e:
        print(f"编码失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_finish_session_encoding()
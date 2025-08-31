import os
import sys

# 设置UTF-8编码环境变量 - 必须在导入任何模块之前设置
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'

# 设置默认编码为UTF-8
if hasattr(sys, 'setdefaultencoding'):
    sys.setdefaultencoding('utf-8')

# 设置文件系统编码
if hasattr(sys, 'setfilesystemencoding'):
    sys.setfilesystemencoding('utf-8')

from chat_engine.chat_engine import ChatEngine
import gradio as gr
import argparse

import gradio
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from loguru import logger

from engine_utils.directory_info import DirectoryInfo
from service.service_utils.logger_utils import config_loggers
from service.service_utils.service_config_loader import load_configs
from service.service_utils.ssl_helpers import create_ssl_context

project_dir = DirectoryInfo.get_project_dir()
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)


def parse_args():
    parser = argparse.ArgumentParser(description='OpenAvatarChat Demo V2 - Manual Recording Control')
    parser.add_argument("--host", type=str, help="service host address")
    parser.add_argument("--port", type=int, help="service host port")
    parser.add_argument("--config", type=str, default="config/chat_with_lam_manual.yaml", help="config file to use")
    parser.add_argument("--env", type=str, default="default", help="environment to use in config file")
    return parser.parse_args()


class OpenAvatarChatWebServer(uvicorn.Server):

    def __init__(self, chat_engine: ChatEngine, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chat_engine = chat_engine
    
    async def shutdown(self, sockets=None):
        logger.info("Start normal shutdown process")
        self.chat_engine.shutdown()
        await super().shutdown(sockets)


def setup_demo():
    app = FastAPI(title="OpenAvatarChat V2", version="2.0.0")

    css = """
    .app {
        @media screen and (max-width: 768px) {
            padding: 8px !important;
        }
    }
    footer {
        display: none !important;
    }
    """
    with gr.Blocks(css=css, title="OpenAvatarChat V2 - Manual Recording") as gradio_block:
        with gr.Column():
            gr.Markdown("# OpenAvatarChat V2 - Manual Recording Control")
            gr.Markdown("## 手动录音控制版本")
            gr.Markdown("点击麦克风按钮开始录音，松开按钮停止录音并发送")
            
            with gr.Group() as rtc_container:
                pass
            
            # 添加手动录音控制说明
            with gr.Row():
                gr.Markdown("""
                ### 使用说明：
                1. 点击"开始对话"按钮建立连接
                2. 按住麦克风按钮进行录音
                3. 松开按钮停止录音并发送语音
                4. 等待AI回复
                """)
    
    gradio.mount_gradio_app(app, gradio_block, "/gradio")
    return app, gradio_block, rtc_container


def main():
    args = parse_args()
    logger_config, service_config, engine_config = load_configs(args)

    # 设置modelscope的默认下载地址
    if not os.path.isabs(engine_config.model_root):
        os.environ['MODELSCOPE_CACHE'] = os.path.join(DirectoryInfo.get_project_dir(),
                                                      engine_config.model_root.replace('models', ''))

    config_loggers(logger_config)
    
    # 禁用VAD自动检测 - 这是关键修改
    if 'SileroVad' in engine_config.handler_configs:
        engine_config.handler_configs['SileroVad']['enabled'] = False
        logger.info("VAD auto-detection disabled for manual recording control")
    
    chat_engine = ChatEngine()
    demo_app, ui, parent_block = setup_demo()

    chat_engine.initialize(engine_config, app=demo_app, ui=ui, parent_block=parent_block)

    ssl_context = create_ssl_context(args, service_config)

    uvicorn_config = uvicorn.Config(demo_app, host=service_config.host, port=service_config.port, **ssl_context)
    server = OpenAvatarChatWebServer(chat_engine, uvicorn_config)
    server.run()


if __name__ == "__main__":
    main()

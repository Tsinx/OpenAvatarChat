import os
import logging
from typing import Optional, Dict, Any
from http import HTTPStatus

import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult

from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDetail, HandlerDataInfo
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.chat_engine_config_data import HandlerBaseConfigModel, ChatEngineConfigModel
from chat_engine.data_models.runtime_data.data_bundle import DataBundleDefinition, DataBundleEntry


class ASRConfig(HandlerBaseConfigModel):
    """ASR配置类"""
    api_key: str = ""
    model_name: str = "paraformer-realtime-v2"
    format: str = "pcm"
    sample_rate: int = 16000
    language_hints: list = ["zh", "en"]


class ASRCallback(RecognitionCallback):
    """ASR回调类"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.result_text = ""
    
    def on_open(self) -> None:
        self.logger.debug('ASR连接已打开')
    
    def on_close(self) -> None:
        self.logger.debug('ASR连接已关闭')
    
    def on_event(self, result: RecognitionResult) -> None:
        sentence = result.get_sentence()
        if sentence:
            self.result_text = sentence
            self.logger.info(f'ASR识别结果: {sentence}')


class ASRContext:
    """ASR上下文类"""
    
    def __init__(self, config: ASRConfig):
        self.config = config
        self.recognition = None
        self.callback = None
        self._setup_recognition()
    
    def _setup_recognition(self):
        """设置识别器"""
        try:
            # 设置API密钥
            dashscope.api_key = self.config.api_key
            
            # 创建回调实例
            self.callback = ASRCallback()
            
            # 创建识别器实例
            self.recognition = Recognition(
                model=self.config.model_name,
                format=self.config.format,
                sample_rate=self.config.sample_rate,
                callback=self.callback
            )
            
        except Exception as e:
            raise RuntimeError(f"初始化ASR识别器失败: {e}")


class HandlerASR(HandlerBase):
    """阿里云百炼ASR处理器"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.asr_context = None
        
        self.logger.info("百炼ASR处理器初始化完成")
    
    def get_handler_info(self) -> HandlerBaseInfo:
        """获取处理器信息"""
        return HandlerBaseInfo(
            config_model=ASRConfig,
            load_priority=0
        )
    
    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[HandlerBaseConfigModel] = None):
        """加载处理器"""
        if handler_config:
            self.asr_context = ASRContext(handler_config)
            self.logger.info(f"百炼ASR处理器加载完成，模型: {handler_config.model_name}")
    
    def create_context(self, session_context: SessionContext, 
                      handler_config: Optional[HandlerBaseConfigModel] = None) -> HandlerContext:
        """创建处理器上下文"""
        return HandlerContext(session_id=session_context.session_info.session_id)
    
    def start_context(self, session_context: SessionContext, handler_context: HandlerContext):
        """启动上下文"""
        pass
    
    def get_handler_detail(self, session_context: SessionContext, 
                          context: HandlerContext) -> HandlerDetail:
        """获取处理器详情"""
        # 定义输出数据格式
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_text_entry("human_text"))
        
        inputs = {
            ChatDataType.HUMAN_AUDIO: HandlerDataInfo(
                type=ChatDataType.HUMAN_AUDIO,
            )
        }
        outputs = {
            ChatDataType.HUMAN_TEXT: HandlerDataInfo(
                type=ChatDataType.HUMAN_TEXT,
                definition=definition,
            )
        }
        return HandlerDetail(
            inputs=inputs,
            outputs=outputs,
        )
    
    def handle(self, context: HandlerContext, inputs: ChatData, 
              output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        """处理音频数据"""
        try:
            if inputs.type != ChatDataType.HUMAN_AUDIO:
                return
            
            # 获取音频数据
            audio_data = inputs.data.get_main_data()
            
            # 调用百炼ASR API进行识别
            if self.asr_context and self.asr_context.recognition:
                result = self.asr_context.recognition.call(audio_data)
                
                if result and hasattr(result, 'output') and result.output:
                    text = result.output.get('sentence', '')
                    if text:
                        # 创建输出数据
                        output_definition = output_definitions.get(ChatDataType.HUMAN_TEXT)
                        if output_definition:
                            from chat_engine.data_models.runtime_data.data_bundle import DataBundle
                            data_bundle = DataBundle(output_definition.definition)
                            data_bundle.set_main_data(text)
                            
                            output_data = ChatData(
                                type=ChatDataType.HUMAN_TEXT,
                                data=data_bundle
                            )
                            
                            # 提交结果
                            context.submit_data(output_data)
                            self.logger.info(f"ASR识别结果: {text}")
                        
        except Exception as e:
            self.logger.error(f"ASR处理失败: {e}")
    
    def destroy_context(self, context: HandlerContext):
        """销毁上下文"""
        pass
    
    def destroy(self):
        """销毁处理器"""
        if self.asr_context:
            self.asr_context = None
        self.logger.info("百炼ASR处理器已销毁")
    

    
    def shutdown(self):
        """关闭处理器"""
        self.logger.info("百炼ASR处理器关闭")
        super().shutdown()
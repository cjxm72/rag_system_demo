"""
配置仅提供默认值；API Key、模型名等由前端 LocalStorage 传入请求体。
"""
import os
import yaml
from pydantic import BaseModel, Field
from typing import Optional

_config = None


class VectorStoreConfig(BaseModel):
    similarity_top_k: int = Field(5, description="检索返回数量")
    rerank_top_n: int = Field(5, description="重排序后保留条数")


class RetrievalConfig(BaseModel):
    hybrid_ratio: float = Field(0.7, description="向量检索权重，1 为纯向量")


class PromptConfig(BaseModel):
    system: str = Field("...", description="系统提示词")
    retrieval: str = Field("...", description="检索模板")


class SiliconFlowConfig(BaseModel):
    api_base: str = Field("https://api.siliconflow.cn/v1", description="API 基础 URL")


class AppConfig(BaseModel):
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    prompts: PromptConfig
    siliconflow: SiliconFlowConfig


def load_config(config_path: str = None) -> AppConfig:
    global _config
    if _config is not None and config_path is None:
        return _config
    if config_path is None:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_file_dir)
        config_path = os.path.join(project_root, "config", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw.setdefault("siliconflow", {"api_base": "https://api.siliconflow.cn/v1"})
    _config = AppConfig(**raw)
    return _config

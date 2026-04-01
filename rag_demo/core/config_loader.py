"""
已废弃：项目不再支持任何“默认配置”（包括默认模型、默认 base_url 等）。

配置必须由前端 LocalStorage 传入请求体 settings（或上传表单字段）提供。
本模块仅为兼容旧 import 路径保留；请不要在业务代码中继续依赖它。
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

_config: Optional["AppConfig"] = None


class VectorStoreConfig(BaseModel):
    similarity_top_k: int = Field(..., description="检索返回数量")
    rerank_top_n: int = Field(..., description="重排序后保留条数")


class RetrievalConfig(BaseModel):
    hybrid_ratio: float = Field(..., description="向量检索权重，1 为纯向量")


class PromptConfig(BaseModel):
    system: str = Field(..., description="系统提示词")
    retrieval: str = Field(..., description="检索模板")


class SiliconFlowConfig(BaseModel):
    api_base: str = Field(..., description="API 基础 URL")


class AppConfig(BaseModel):
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    prompts: PromptConfig
    siliconflow: SiliconFlowConfig


def load_config(config_path: str | None = None) -> AppConfig:
    global _config
    # 兼容旧接口：忽略 config_path（已不再读取任何文件）
    if _config is not None:
        return _config

    # 彻底禁止“默认配置”：任何调用都应显式传入配置，不应依赖 load_config。
    raise RuntimeError("config_loader 已废弃：请从请求 settings/表单字段显式传入配置，不支持默认配置")
    return _config


import yaml
from pydantic import BaseModel,Field
from typing import Dict
import os


_config = None

class ModelConfig(BaseModel):
    name: str = Field(..., description="模型名称")
    provider: str = Field(..., description="模型提供商")
    api_key: str = Field("", description="API访问密钥")
    model_path: str = Field("", description="本地模型路径")

class VectorStoreConfig(BaseModel):
    index_path: str = Field(..., description="FAISS索引存储路径")
    similarity_top_k: int = Field(5, description="相似度检索返回数量")

class RetrievalConfig(BaseModel):
    hybrid_ratio: float = Field(0.7, description="混合检索权重比例")
    rerank_model: str = Field(..., description="重排序模型名称")
    model_path: str = Field("", description="本地模型路径")

class PromptConfig(BaseModel):
    system: str = Field(..., description="系统角色提示词")
    retrieval: str = Field(..., description="检索增强生成提示模板")


class AppConfig(BaseModel):
    models: Dict[str, ModelConfig]
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    prompts: PromptConfig


def load_config(config_path: str = None) :
    if config_path is None:
        global _config
        if _config is not None:
            return _config
            # 获取当前文件（config_loader.py）的绝对路径
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        # 计算项目根目录
        project_root = os.path.dirname(current_file_dir)
        # 拼接配置文件路径
        config_path = os.path.join(project_root, "config", "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
    _config = AppConfig(**raw_config)
    return _config


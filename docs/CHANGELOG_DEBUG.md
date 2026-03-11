# 修改记录与错误修复日志（毕设用）

本文件用于记录本项目在重构过程中的关键改动、出现的错误以及对应修复，便于毕设写作与答辩说明。

## 目标与大方向

- 目标：FastAPI + RAG Demo，**全面转向在线 API**（硅基流动为主），本地仅保留 Ollama（OpenAI 兼容接口）作为“可本地运行不泄露数据”的噱头与兜底。
- 向量库：全局单索引（FAISS），业务侧维护“知识组 – 文档映射”，支持多组选中与主/次组加权。
- 配置：前端 `LocalStorage` 保存 API Key 与模型名，后端不保存密钥。

---

## 重大改动（按模块）

### 1) 模型与依赖结构

- 删除 `server/`、`models/`：不再依赖本地 llama-server、本地 embedding/reranker 模型文件。
- LLM：
  - 硅基流动：在线 ChatCompletions；
  - Ollama：仅通过 OpenAI 兼容 `base_url` 调用（答辩可演示“本地也能跑”）。
- Embedding / Reranker：
  - 走硅基流动在线接口（`/embeddings`、`/rerank`）。

### 2) RAG 与知识组

- 文档表：每个文档只保存 **唯一 ID + 原始内容**（以及文件名/路径等元信息），不绑定任何组。
- 知识组表：仅保存组与文档的关联（doc_ids），支持一个文档属于多个组。
- 向量库：FAISS 单索引，仅存储 chunk 向量，chunk metadata 中只保留 `doc_id`。
- 查询：支持一次选择多个知识组，每个组带 `priority`（主/次组权重），在召回结果上进行加权后再 rerank。

### 3) Agent 记忆与新对话

- 使用 LangGraph 的 `MemorySaver()`，按 `thread_id` 分离记忆。
- 前端提供“新对话”按钮：生成新的 `thread_id`，从而开启新会话记忆。

### 4) 前端

- 主页面：左侧对话；右侧放“上传文档 + 知识组选择/管理”。
- 设置页：API Key、模型名等写入 LocalStorage；每次请求 body 携带 settings。

### 5) 文档解析

- PDF：PyMuPDF 提取文本。
- 图片：硅基视觉模型解析（图 → 文）。
- 上传：支持多文件选择，前端逐个上传；后端也支持一次请求传多个文件字段。

---

## 错误与修复记录

### 错误 1：`'Qwen/Qwen3-Embedding-0.6B' is not a valid OpenAIEmbeddingModelType`

**原因**：LlamaIndex 的 `OpenAIEmbedding` 在部分版本中对模型名做枚举校验，硅基模型名不在其枚举里。  
**修复**：实现自定义 `SiliconFlowEmbedding`（直接调用硅基 `/embeddings`），不再使用 `OpenAIEmbedding` 的枚举校验。

涉及文件：
- `src/siliconflow_embedding.py`（新增）
- `src/rag_system.py`（替换 embedding 实现）

### 错误 2：`TypeError: Can't instantiate abstract class SiliconFlowEmbedding without ... _aget_query_embedding`

**现象**：RAG 提问时在 `retrieve` 节点报错，提示缺少异步抽象方法。  
**原因**：`BaseEmbedding` 需要异步抽象方法（如 `_aget_query_embedding`），初版仅实现了同步方法。  
**修复**：
- 为 `SiliconFlowEmbedding` 补齐 async 方法：`_aget_query_embedding` / `_aget_text_embedding` / `_aget_text_embeddings`（使用 `httpx.AsyncClient`）。
- 为避免运行时加载到旧模块，统一导入路径为 `from src.xxx import ...` 并新增 `src/__init__.py`，减少模块名冲突与缓存导致的“看似已修复但仍报错”的情况。

涉及文件：
- `src/siliconflow_embedding.py`（补齐 async）
- `src/__init__.py`（新增）
- `src/main.py`、`src/agent.py`、`src/rag_system.py`、`src/eval_rag.py`（统一 `src.*` 导入）

### 错误 3：`ValueError: "SiliconFlowEmbedding" object has no field "api_key"`

**现象**：检索阶段报错，提示 embedding 对象没有字段 `api_key`。  
**原因**：`BaseEmbedding` 继承自 Pydantic 模型，默认不允许动态添加未声明字段；初版在 `__init__` 中写了 `self.api_key = ...` 等动态属性。  
**修复**：将 `api_key/api_base/model/timeout` 等改为 Pydantic 的私有属性（`PrivateAttr`），使用 `self._api_key` 等存储。

涉及文件：
- `src/siliconflow_embedding.py`（引入 `PrivateAttr` 并改用 `_api_key/_api_base/_model/_timeout_s`）

---

## 评估体系（RAG Evaluation）

新增接口：`POST /evaluate`

思路：轻量且适合毕设展示（不引入复杂评测框架）：
- 对每条样例：跑一次检索 + 生成；
- 输出每题：question / expected / answer / context；
- 用同一 embedding 模型计算 `answer` vs `expected` 的余弦相似度作为“语义相关性”指标；
- 汇总输出平均分，便于写“改进前后对比”“不同知识组策略对比”等实验。

涉及文件：
- `src/eval_rag.py`（新增）
- `src/main.py`（新增 `/evaluate` 路由）


# 架构与实现说明（当前版本）

本文档用于毕设写作：解释本项目当前的总体架构、数据结构、RAG 检索流程、知识组权重、多轮记忆以及评估体系。

## 1. 总体架构

### 1.1 组件划分

- **前端（静态）**：`static/index.html`（主页面）、`static/settings.html`（设置页）
  - 配置（API Key、模型名、Ollama URL 等）写入 **LocalStorage**；
  - 每次请求将 `settings`（以及 `groups`）放入请求体，不在后端保存密钥。
- **后端（FastAPI）**：`src/main.py`
  - 文档上传与解析、知识组管理、RAG 问答、多轮会话记忆、评估接口。
- **文档/知识组存储（业务表）**：`src/store.py`
  - 单文件：`data/store.json`
- **向量库（FAISS）**：`src/rag_system.py`
  - 全局 **单索引**；
  - 向量库只感知 `doc_id`（chunk metadata）与向量，不感知“组”。
- **Embedding（在线）**：`src/siliconflow_embedding.py`
  - 直接调用硅基流动 `/embeddings`，支持任意硅基模型名（不受枚举限制）。
- **Reranker（在线）**：`src/siliconflow_rerank.py`
  - 调用硅基流动 `/rerank` 做精排。
- **视觉解析（在线）**：`src/vision_api.py`
  - 图片提问与图片文档解析使用硅基视觉模型。
- **PDF 解析**：`src/doc_parser.py`（PyMuPDF 提取文本）
- **Agent 编排与记忆**：`src/agent.py`（LangGraph + MemorySaver）
- **评估体系**：`src/eval_rag.py` + `POST /evaluate`

---

## 2. 数据结构设计

### 2.1 文档（Documents）

每个文档只保留“唯一 ID + 原始内容（解析后的文本）”，不绑定任何组信息：

```json
{
  "id": "uuid",
  "name": "原文件名.pdf",
  "path": "data/uploads/xxx.pdf",
  "text": "解析后的纯文本..."
}
```

### 2.2 知识组（Groups）

知识组是业务侧概念，仅保存组与文档的映射：

```json
{
  "id": "uuid",
  "name": "财务制度",
  "doc_ids": ["doc_uuid_1", "doc_uuid_2"]
}
```

> 注意：**向量库不存 group_id**。知识组信息只在检索阶段生效。

---

## 3. 向量库与检索流程（单索引 + 业务侧分组）

### 3.1 单索引策略

所有文档切 chunk 后统一写入同一个 FAISS 索引。每个 chunk 的 metadata 至少包含：

- `doc_id`：所属文档 ID

### 3.2 多组选择 + 主/次组权重

前端在一次提问中可选择多个知识组，并为每个组设置优先级：

```json
"groups": [
  { "id": "groupA", "priority": 1.0 },   // 主组
  { "id": "groupB", "priority": 0.5 }    // 次要
]
```

后端检索流程：

1. **全库向量召回**：取 topK 候选 chunk；
2. **业务侧过滤**：仅保留 `doc_id` 属于被选中组的候选；
3. **组权重加权**：对每个候选按 `doc_id` 获取权重 \(w_{doc}\)，更新分数：

\[
score' = score \times (1 + \lambda \cdot w_{doc})
\]

4. **Reranker 精排**：将候选文本交给硅基 `/rerank`，取 top_n 作为上下文；
5. **LLM 生成**：将上下文与问题组合为 system prompt，让 LLM 回答。

> 这个设计满足：文档可属于多个组；一次查询可选多个组；主/次组只影响排序，不需要向量库感知组。

---

## 4. 多轮记忆与“新对话”

后端使用 LangGraph 的 `MemorySaver()` 按 `thread_id` 记录会话消息。

- 同一 `thread_id`：多轮对话会记忆上下文；
- 点击前端“新对话”：生成新的 `thread_id`（如 `thread-时间戳`），从而开启一段新的会话记忆。

---

## 5. 接口清单（FastAPI）

- `GET /`：主页面
- `GET /settings`：设置页
- `POST /query`：问答（支持 `groups`、`settings`、可选 `image_base64`）
- `GET /documents`：文档列表
- `POST /documents/upload`：上传文档（支持多文件）
- `DELETE /documents/{doc_id}`：删除文档
- `GET /groups`：知识组列表
- `POST /groups`：新建知识组
- `PUT /groups/{group_id}`：更新组（doc_ids）
- `DELETE /groups/{group_id}`：删除组
- `POST /evaluate`：评估（批量样例）

---

## 6. 评估体系（当前实现）

`/evaluate` 对每条样例输出：

- `context`：检索到的上下文（可用于人工检查）
- `answer`：模型答案
- `semantic_similarity`：`answer` 与 `expected` 的 embedding 余弦相似度（语义相关性 proxy）

适合毕设实验：

- 对比不同 embedding 模型 / reranker 模型；
- 对比“单组选中”与“多组加权”；
- 对比不同 chunk_size/topK/rerank_n 配置。


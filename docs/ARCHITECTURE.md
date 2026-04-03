# 架构与实现说明（当前版本）

本文档说明：总体架构、PostgreSQL（SQLModel）与 Milvus、检索与引用、对话记忆、PDF 解析、Agno Team 与接口。

## 1. 目录结构（核心代码）

| 路径 | 职责 |
|------|------|
| `rag_demo/api/main.py` | FastAPI：上传、知识组、`POST /query`（SSE）、评估 |
| `rag_demo/core/provider_settings.py` | 硅基 / OpenAI / Ollama 的 Base、Key 解析 |
| `rag_demo/api/sse.py` | SSE 帧编码 |
| `rag_demo/storage/models.py` | SQLModel 表：`Document`、`Group`、`GroupMember`、`ChatMessage`、`AppMeta` |
| `rag_demo/storage/database.py` | PostgreSQL 引擎、`create_db_and_tables` |
| `rag_demo/storage/db.py` | 业务 CRUD（与旧 SQLite 版函数签名一致） |
| `rag_demo/storage/store.py` | 对上层导出与 `db` 一致的 API |
| `rag_demo/rag/rag_system.py` | Milvus 索引 + 检索、rerank、上下文拼装、`format_citation_footer` |
| `rag_demo/rag/milvus_store.py` | Milvus collection `rag_kb` 封装 |
| `rag_demo/rag/types.py` | `RAGResult` / `RetrievedChunk` |
| `rag_demo/rag/siliconflow_*.py` | OpenAI 兼容 Embedding；硅基风格 Rerank（`/rerank`） |
| `rag_demo/parsing/doc_parser.py` | PDF（OpenDataLoader v2+）/ DOCX / 图片 / 文本 |
| `rag_demo/agents/agno_team.py` | Agno Team：医疗、法律、协调者 |
| `rag_demo/core/singleton.py` | `@singleton` 模型与 Team 工厂缓存 |
| `static/` | 前端静态页 |

数据与外部服务：

- **PostgreSQL**：`DATABASE_URL` 或 `.env` 中 `POSTGRES_*`（默认 `127.0.0.1:5432`、库 `postgres`、用户 `postgres`、密码 `localhost`、schema `public`）
- **Milvus**：`MILVUS_URI` 或 `MILVUS_HOST`/`MILVUS_PORT`，collection 名默认 `rag_kb`（可用 `MILVUS_COLLECTION` 覆盖）
- `data/uploads/`：上传原文件（相对 `project_root`）
- `data/store.json`：仅**迁移源**（若存在且库为空则导入一次）

**Milvus** 与 **PostgreSQL** 均按本机部署配置（见 `.env.example`），仓库不再附带 Docker Compose。

## 2. PostgreSQL 表设计（与旧版语义一致）

- **documents**：`id, name, path, text, status, progress, error, created_at` — 文档正文与元数据，供引用反查。
- **groups** + **group_members**：知识组与多对多文档关系。
- **chat_messages**：`thread_id, role, content, created_at` — 服务端多轮记忆。
- **app_meta**：如 `embedding_model`、`milvus_built_embedding`（用于判断是否需要重建向量索引；兼容读取旧键 `chroma_built_embedding`）。

表由 **SQLModel** 定义，`init_db()` 时 `create_all` 建表。

## 3. 向量库（Milvus）

- 单 collection（默认 `rag_kb`），字段：`pk, doc_id, chunk_index, name, text, embedding`。
- 文档变更后 `ensure_index(..., force_rebuild=True)` 会 **drop 并重建** collection，再全量写入 chunk 与向量。
- 若 **embedding 模型**变更，通过 `app_meta` 比对触发重建。

## 4. 模型接入（硅基 / OpenAI / Ollama）

- **Embedding / 视觉 / LLM**：均为 OpenAI 兼容 HTTP（`base_url` + `api_key`）；Ollama 使用 `http://127.0.0.1:11434/v1` 等，Key 可填任意占位。
- **Rerank**：仅 **硅基流动**（及兼容 `/v1/rerank` 的网关）走在线 rerank；选 OpenAI 或 Ollama 时按向量分数截断，不调用 rerank 接口。

## 5. 检索与引用（非仅提示词）

1. 向量召回（Milvus）→ 组过滤（`doc_id in [...]` 表达式）→ 组权重 →（可选）硅基 rerank。
2. 每条片段写入 `RetrievedChunk`（含 `doc_id`、`chunk_index`、`source_id`）。
3. 拼入 LLM 的上下文字符串中每条带：`[来源i] doc_id=… 文档名=…（来自 PostgreSQL 反查）chunk=…`。
4. 回答结束后，后端根据本轮 **去重后的 doc_id 列表** 再次 `get_documents_by_ids`，生成文末固定块：

```text
---
【引用文档】
1. 《文件名》  doc_id=uuid
```

5. SSE 事件中 `type: sources` 载荷为 `[{doc_id, name}]`，便于前端展示。

## 6. 对话记忆

- 每次提问前：从 `chat_messages` 读取该 `thread_id` 的历史，拼成「历史对话」再与当前用户输入组合后送入检索与生成。
- 若库中尚无记录，可回退使用请求体中的 `history`（兼容旧前端）。
- 用户消息与助手完整回复（含路由说明与引用块）写入 PostgreSQL。

## 7. PDF / 文档解析

- **PDF**：优先 `opendataloader_pdf.convert(..., format="text")`，输出目录下读取 `.txt/.md`；异常或空结果则 **PyMuPDF** 回退。
- **DOCX**：`python-docx` 读段落文本。
- **图片**：OpenAI 兼容多模态 Chat（硅基 / OpenAI / Ollama，取决于设置中的 vision 提供商）。

**注意**：OpenDataLoader 官方要求本机安装 **Java 11+**。

## 8. Agno Team

- **Medical / Legal Agent**：提示词模拟领域行为。
- **Coordinator**：`TeamMode.coordinate`，由协调者模型委派或融合；成员与协调者可配置不同 `OpenAILike` 模型 id。
- 路由由协调者模型输出 JSON（领域 + delegates），再决定调用单专家或 Team 协调。

## 9. 接口摘要

- `GET /`、`GET /settings`：静态页  
- `POST /query`：**唯一问答出口**，`text/event-stream`（SSE）；事件类型含 `route`、`chunk`、`sources`、`done`；正文仍含「路由说明」前缀与文末「引用文档」块（由 chunk 拼接）  
- `GET/POST/PUT/DELETE`：`/documents`、`/groups`  
- `POST /evaluate`：评估  

## 10. 依赖与启动

- 安装：`uv sync`
- 配置：根目录 `.env`（或环境变量）`DATABASE_URL` / `POSTGRES_*`、`MILVUS_URI`（或 Host/Port）
- 启动：`uv run python -m rag_demo` 或 `uvicorn rag_demo.api.main:app --port 8001`

已移除旧版 `src/` 兼容层；入口统一为包 `rag_demo`。

## 11. 自动化测试

- 说明与命令见 **`docs/TESTING.md`**。
- 集成测试需 **PostgreSQL + Milvus**；`tests/conftest.py` 会在用例间 **truncate PG 表并 drop Milvus collection**，上传目录使用临时目录。

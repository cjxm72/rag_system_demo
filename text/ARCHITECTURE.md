# 架构与实现说明（当前版本）

本文档说明：总体架构、SQLite 与 Chroma、检索与引用、对话记忆、PDF 解析、Agno Team 与接口。

## 1. 目录结构（核心代码）

| 路径 | 职责 |
|------|------|
| `rag_demo/api/main.py` | FastAPI：上传、知识组、`/query`、`/query/stream`、评估 |
| `rag_demo/storage/db.py` | SQLite：`documents`、`groups`、`group_members`、`chat_messages`、`app_meta` |
| `rag_demo/storage/store.py` | 对上层导出与 `db` 一致的 API |
| `rag_demo/rag/rag_system.py` | Chroma + 检索、rerank、上下文拼装、`format_citation_footer` |
| `rag_demo/rag/types.py` | `RAGResult` / `RetrievedChunk` |
| `rag_demo/rag/siliconflow_*.py` | 硅基 Embedding / Rerank |
| `rag_demo/parsing/doc_parser.py` | PDF（OpenDataLoader v2+）/ DOCX / 图片 / 文本 |
| `rag_demo/agents/agno_team.py` | Agno Team：医疗、法律、协调者 |
| `rag_demo/core/singleton.py` | `@singleton` 模型与 Team 工厂缓存 |
| `static/` | 前端静态页 |

数据目录：

- `data/app.db`：SQLite
- `data/chroma/`：Chroma 持久化
- `data/uploads/`：上传原文件
- `data/store.json`：仅**迁移源**（若存在且库为空则导入一次）

## 2. SQLite 表设计

- **documents**：`id, name, path, text, status, progress, error` — 文档正文与元数据，供引用反查。
- **groups** + **group_members**：知识组与多对多文档关系。
- **chat_messages**：`thread_id, role, content` — 服务端多轮记忆。
- **app_meta**：如 `embedding_model`、`chroma_built_embedding`（用于判断是否需要重建向量索引）。

## 3. 向量库（Chroma）

- 使用 `chromadb.PersistentClient(path=data/chroma)`，单 collection `rag_kb`。
- 文档变更后 `ensure_index(..., force_rebuild=True)` 会删除并重建该 collection（实现简单、一致性强）。
- 若 **embedding 模型**变更，通过 `app_meta` 比对触发重建。

## 4. 检索与引用（非仅提示词）

1. 向量召回 → 组过滤 → 组权重 → 硅基 rerank。
2. 每条片段写入 `RetrievedChunk`（含 `doc_id`、`chunk_index`、`source_id`）。
3. 拼入 LLM 的上下文字符串中每条带：`[来源i] doc_id=… 文档名=…（来自 SQLite 反查）chunk=…`。
4. 回答结束后，后端根据本轮 **去重后的 doc_id 列表** 再次 `get_documents_by_ids`，生成文末固定块：

```text
---
【引用文档】
1. 《文件名》  doc_id=uuid
```

5. `POST /query` 的 JSON 另含 `sources: [{doc_id, name}]`，便于前端展示。

## 5. 对话记忆

- 每次提问前：从 `chat_messages` 读取该 `thread_id` 的历史，拼成「历史对话」再与当前用户输入组合后送入检索与生成。
- 若库中尚无记录，可回退使用请求体中的 `history`（兼容旧前端）。
- 用户消息与助手完整回复（含路由说明与引用块）写入 SQLite。

## 6. PDF / 文档解析

- **PDF**：优先 `opendataloader_pdf.convert(..., format="text")`，输出目录下读取 `.txt/.md`；异常或空结果则 **PyMuPDF** 回退。
- **DOCX**：`python-docx` 读段落文本。
- **图片**：硅基视觉 API（与原先一致）。

**注意**：OpenDataLoader 官方要求本机安装 **Java 11+**。

## 7. Agno Team

- **Medical / Legal Agent**：提示词模拟领域行为。
- **Coordinator**：`TeamMode.coordinate`，由协调者模型委派或融合；成员与协调者可配置不同 `OpenAILike` 模型 id。
- 用户问题先经简单关键词路由提示领域；协调者可在 Team 内进一步分配任务。

## 8. 接口摘要

- `GET /`、`GET /settings`：静态页  
- `POST /query`：JSON，`QueryResponse` 含 `answer`、`thread_id`、`sources`  
- `POST /query/stream`：`text/plain` 流式；末尾追加引用块；路由说明在流开头  
- `GET/POST/PUT/DELETE`：`/documents`、`/groups`  
- `POST /evaluate`：评估  

## 9. 依赖与启动

- 安装：`uv sync`
- 启动：`uv run python -m rag_demo` 或 `uvicorn rag_demo.api.main:app --port 8001`

已移除旧版 `src/` 兼容层；入口统一为包 `rag_demo`。

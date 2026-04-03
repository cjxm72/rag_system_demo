# 评估（用于论文的明细指标）

本项目提供 `POST /evaluate` 用于对 RAG 的 **检索+生成** 进行评估，并返回可直接用于论文写作的明细数据（chunk、分数、引用、耗时等）。

## 1. 评估输入

请求体：

- `items[]`：
  - `question`：问题
  - `expected`：期望答案（用于语义相似度 proxy）
  - `groups`：知识组选择（同 `/query` 的 `groups`）
- `settings`：与 `/query` 一致（来自前端 LocalStorage），至少需要：
  - `api_key`、`api_base`
  - `embedding_model`、`reranker_model`
  - `llm_provider`、`llm_model`
  - `temperature`、`max_tokens`

## 2. 评估输出字段说明

`items[i]` 中的关键字段：

- `answer`：LLM 实际回答
- `context`：拼给 LLM 的检索上下文（含 `[来源i] doc_id=... chunk=...`）
- `semantic_similarity`：`cos(answer_embedding, expected_embedding)`（用于粗略对齐度）

### 2.1 Retrieval 明细（论文关键）

`items[i].retrieval`：

- `similarity_top_k`：向量召回数量
- `rerank_top_n`：重排保留数量
- `citation_doc_ids`：本轮引用涉及的 doc_id（用于文末 `【引用文档】`）
- `chunks[]`：每个检索片段
  - `source_id`：上下文编号（对应 `[来源{source_id}]`）
  - `doc_id`：文档 id（PostgreSQL `documents` 表）
  - `chunk_index`：文档内 chunk 序号
  - `score`：（重排后）分数
  - `text`：chunk 原文（可用于论文附录/案例分析）

### 2.2 Timing / Stats（论文可用）

- `timing_s`：拆分耗时（retrieve / llm / embeddings_for_metrics / total）
- `stats`：上下文/回答长度、chunk 数、引用文档数等

## 3. 运行示例（pytest）

集成测试覆盖 `/evaluate`：`tests/test_rag_evaluate.py`。
运行：

```bash
cd /path/to/rag_system_demo
uv sync --group dev
uv run pytest tests/ -m integration -v --tb=short
```


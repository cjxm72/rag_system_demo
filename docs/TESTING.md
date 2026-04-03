# 测试说明

## 依赖

- 开发依赖含 `pytest`：`uv sync --group dev`
- **集成测试**需要：本机 **PostgreSQL**、本机 **Milvus**、网络与 **API Key**（嵌入、rerank、LLM）。请自行安装并启动服务，环境变量见 `tests/test.env.example` 与项目根 `.env.example`。
- 在终端查看 **SSE 聚合后的回答摘要**：`RAG_TEST_PRINT_ANSWERS=1 uv run pytest tests/ -v -s`（`-s` 关闭输出捕获）。

## 环境变量

| 变量 | 说明 |
|------|------|
| `DATABASE_URL` | PostgreSQL 连接串，如 `postgresql+psycopg://user:pass@127.0.0.1:5432/dbname`。未设置时，依赖 `client` / `isolated_env` 的用例会跳过；仅需 PG 的用例可使用 `bare_client`。 |
| `MILVUS_URI` 或 `MILVUS_HOST` + `MILVUS_PORT` | Milvus 连接。未设置时，完整 RAG 集成用例会跳过。可选 `MILVUS_COLLECTION` 覆盖默认 collection 名。 |
| `SILICONFLOW_API_KEY` / `RAG_DEMO_API_KEY` / `OPENAI_API_KEY` | 集成测试必需；未设置（或值为 `xx`）时跳过集成用例。 |
| `RAG_TEST_API_BASE` 或 `OPENAI_BASE_URL` | 可选，默认 `https://api.siliconflow.cn/v1`。 |
| `RAG_TEST_LLM` 或 `OPENAI_MODEL` | 可选，覆盖默认 LLM 模型 id。 |
| `RAG_TEST_EMBEDDING` 或 `EMBEDDING_MODEL` | 可选，覆盖嵌入模型。 |
| `RAG_TEST_RERANKER` 或 `RERANKER_MODEL` | 可选，覆盖 rerank 模型。 |
| `RAG_TEST_VISION` 或 `OCR_MODEL` | 可选，覆盖视觉模型（上传含图档时）。 |

## 命令

```bash
cd /path/to/rag_system_demo
uv sync --group dev

# 仅本地、不连外网（需配置 DATABASE_URL；健康检查等）
uv run pytest tests/ -m "not integration" -v

# 完整集成（需 Key）
set -a
source tests/test.env
set +a
uv run pytest tests/ -m integration -v --tb=short
```

注意：如果你只在当前终端写了 `OPENAI_API_KEY=...` 但**没有 export**，子进程（pytest）是拿不到的。
务必使用 `export OPENAI_API_KEY=...`，或像上面一样用 `set -a; source ...` 自动导出。

## 测试数据

- `tests/text/`：`legal_sample.txt`、`medical_sample.txt`、`mixed_sample.txt`，用于关键词路由与检索。
- PDF 样例由测试内 **PyMuPDF** 动态生成；若本机已装 **Java 11+**，上传 PDF 时会走 OpenDataLoader PDF v2 解析路径（与生产一致）。

## 覆盖范围

| 文件 | 内容 |
|------|------|
| `tests/test_e2e_rag.py` | 隔离 PostgreSQL 与 Milvus 下的端到端：上传 TXT/PDF、建知识组、`POST /query`（SSE）、医疗/法律/综合路由、`【引用文档】` 与 `sources`、三模型字段（医疗/法律/协调者）。 |
| `tests/test_rag_evaluate.py` | `POST /evaluate`：条目结构、`semantic_similarity` 与聚合指标。 |

## Marker

- `integration`：需要 Key 与网络。
- `slow`：含嵌入或大模型，耗时较长。

配置见根目录 `pyproject.toml` 中 `[tool.pytest.ini_options]`。

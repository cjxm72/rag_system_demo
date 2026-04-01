# 测试说明

## 依赖

- 开发依赖含 `pytest`：`uv sync --group dev`
- **集成测试**需要网络与 **硅基流动 API Key**（嵌入、rerank、LLM）。

## 环境变量

| 变量 | 说明 |
|------|------|
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

# 仅本地、不连外网（当前为健康检查等）
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
| `tests/test_e2e_rag.py` | 隔离库与 Chroma 下的端到端：上传 TXT/PDF、建知识组、`/query` 与 `/query/stream`、医疗/法律/综合路由、`【引用文档】` 与 `sources`、三模型字段（医疗/法律/协调者）。 |
| `tests/test_rag_evaluate.py` | `POST /evaluate`：条目结构、`semantic_similarity` 与聚合指标。 |

## Marker

- `integration`：需要 Key 与网络。
- `slow`：含嵌入或大模型，耗时较长。

配置见根目录 `pyproject.toml` 中 `[tool.pytest.ini_options]`。

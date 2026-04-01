# 企业数据管理秘书 · RAG（Chroma + SQLite + 全在线）

这是一个用于毕设展示的 RAG Demo，核心特点：

- **向量库 Chroma**：持久化目录 `data/chroma/`，单 collection；chunk metadata 含 `doc_id`，知识组仍在业务侧映射。
- **结构化存储 SQLite**：文档/知识组/对话记忆存于 `data/app.db`（无需单独安装数据库服务）；首次启动可从旧版 `data/store.json` 自动迁移。
- **全在线**：Embedding / Reranker / LLM / 图片解析均可通过 **硅基流动 API** 完成。
- **Ollama 兜底**：LLM 可切换为 Ollama（OpenAI 兼容接口），用于答辩说明本地部署可行性。
- **PDF 解析 OpenDataLoader PDF v2+**：优先使用 `opendataloader-pdf`（需系统安装 **Java 11+**）；失败时回退 PyMuPDF。
- **Office**：支持 `.docx`（`python-docx`）。
- **引用可追溯**：检索阶段保留 `doc_id`，上下文带 `[来源i]`；回答末尾由后端根据 SQLite **反查文档名** 追加 `【引用文档】` 列表；`POST /query` 另返回 `sources` 字段。
- **多 Agent（Agno Team）**：医疗 / 法律 / 协调者；协调者路由并委派成员模型。
- **前端保存密钥**：API Key 与模型名在 **LocalStorage**，请求体 `settings` 传入后端。

---

## 功能

- **对话**：服务端按 `thread_id` 将多轮消息写入 SQLite；前端仍可传 `history` 作为**空库时**的兜底。
- **文档上传**：多文件；PDF / DOCX / 文本 / 图片等。
- **知识组**：组-文档映射在 SQLite；检索时按组过滤与加权 + 硅基 rerank。
- **评估**：`POST /evaluate`。
- **设置页**：可分别配置默认 LLM 与医疗/法律/协调者模型 id。

---

## 快速开始

```bash
cd /path/to/rag_system_demo
uv sync
uv run python -m rag_demo
```

或：

```bash
uv run uvicorn rag_demo.api.main:app --host 0.0.0.0 --port 8001
```

打开：

- 主页面：`http://127.0.0.1:8001/`
- 设置页：`http://127.0.0.1:8001/settings`
- API 文档：`http://127.0.0.1:8001/docs`

### OpenDataLoader PDF（可选但推荐）

- 安装 **Java 11+** 并确保 `java -version` 可用。
- Python 依赖已包含 `opendataloader-pdf`；若未装 Java，PDF 将自动回退 PyMuPDF。

---

## 关键配置（前端设置页）

- Embedding / Reranker / 默认 LLM / 视觉模型：见设置页占位符。
- 可选：`llm_model_medical`、`llm_model_legal`、`llm_model_coordinator`（不填则与默认 `llm_model` 相同）。

---

## 文档

- 架构与实现：`docs/ARCHITECTURE.md`
- 毕设写作素材：`docs/THESIS_WRITING.md`
- 调试记录：`docs/CHANGELOG_DEBUG.md`
- **测试说明（pytest、环境变量、集成范围）**：`docs/TESTING.md`
- **评估说明（论文用明细指标）**：`docs/EVALUATION.md`

---

## 运行测试

```bash
uv sync --group dev
# 不连外网（健康检查等）
uv run pytest tests/ -m "not integration" -v
# 完整集成：把 tests/test.env 里的 OPENAI_API_KEY 改成你的 Key，然后运行（详见 docs/TESTING.md）
set -a
source tests/test.env
set +a
uv run pytest tests/ -m integration -v
```

---

## 常见问题排查

（真实故障可记入 `docs/CHANGELOG_DEBUG.md`。）

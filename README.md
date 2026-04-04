# 企业数据管理秘书 · RAG（Milvus + PostgreSQL + 全在线）

这是一个用于毕设展示的 RAG Demo，核心特点：

- **向量库 Milvus**：独立向量服务，单 collection（默认 `rag_kb`）；chunk 含 `doc_id` 等元数据，知识组仍在业务侧映射。
- **结构化存储 PostgreSQL**：文档/知识组/对话记忆由 **SQLModel** 建表；通过根目录 **`.env`** 中 `DATABASE_URL` 或 `POSTGRES_*` 连接（默认本机 `postgres` 库、`public` schema）。
- **多提供商**：Embedding / Rerank / LLM / 视觉 均可独立选择 **硅基流动**、**OpenAI** 或 **Ollama**（OpenAI 兼容 HTTP）；Rerank 在 OpenAI/Ollama 下按向量分截断（无标准 rerank 接口）。
- **问答输出**：仅 **`POST /query` SSE**（`text/event-stream`），不再提供 JSON 整包问答接口。
- **PDF 解析 OpenDataLoader PDF v2+**：优先使用 `opendataloader-pdf`（需系统安装 **Java 11+**）；失败时回退 PyMuPDF。
- **Office**：支持 `.docx`（`python-docx`）。
- **引用可追溯**：检索阶段保留 `doc_id`，上下文带 `[来源i]`；回答末尾由后端根据 PostgreSQL **反查文档名** 追加 `【引用文档】` 列表；SSE 流中 `sources` 事件返回引用列表。
- **多 Agent（Agno Team）**：医疗 / 法律 / 协调者；协调者路由并委派成员模型。
- **前端保存密钥**：API Key 与模型名在 **LocalStorage**，请求体 `settings` 传入后端。

---

## 功能

- **对话**：服务端按 `thread_id` 将多轮消息写入 PostgreSQL；前端仍可传 `history` 作为**空库时**的兜底。
- **文档上传**：多文件；PDF / DOCX / 文本 / 图片等。
- **知识组**：组-文档映射在 PostgreSQL；检索时按组过滤与加权 + 硅基 rerank。
- **评估**：`POST /evaluate`。
- **设置页**：分别为 Embedding / Rerank / LLM / 视觉 选择硅基、OpenAI 或 Ollama，并可填专用 Base、Key；另可配置医疗/法律/协调者模型 id。

---

## 快速开始

准备环境变量（推荐复制 `.env.example` 为 `.env`；**PostgreSQL 与 Milvus 均使用本机服务**，不再提供 Docker Compose）：

```bash
cd /path/to/rag_system_demo
cp .env.example .env
# 本机需已启动 PostgreSQL（默认见 .env.example）与 Milvus standalone（如 127.0.0.1:19530）
```

再安装依赖并启动应用：

```bash
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
- 可选：`llm_model_medical`、`llm_model_legal`（不填则与默认 `llm_model` 相同）。协调者路由与默认回答共用 `llm_model`。
- Rerank：硅基走 `/rerank`；选 OpenAI 或 Ollama 时用 OpenAI 兼容 `/embeddings` 做余弦相似度重排（`reranker_model` 填 **embedding 模型 id**）。

---

## 文档

- 架构与实现：`docs/架构与实现说明.md`
- 毕设写作素材：`docs/论文写作素材.md`
- 调试记录：`docs/修改记录与错误修复日志.md`
- **测试说明（pytest、环境变量、集成范围）**：`docs/测试说明.md`
- **评估说明（论文用明细指标）**：`docs/评估说明.md`
- **目录说明 + 建表 SQL + 索引（论文用）**：`docs/目录结构说明（论文用）.md`
- **论文修改及流程图生成（按当前项目实现）**：`docs/论文修改及流程图生成.md`

---

## 运行测试

```bash
uv sync --group dev
# 不连外网（健康检查等）
uv run pytest tests/ -m "not integration" -v
# 完整集成：把 tests/test.env 里的 OPENAI_API_KEY 改成你的 Key，然后运行（详见 docs/测试说明.md）
set -a
source tests/test.env
set +a
uv run pytest tests/ -m integration -v
```

---

## 常见问题排查

（真实故障可记入 `docs/修改记录与错误修复日志.md`。）

# 企业数据管理秘书 · RAG（全在线 + Ollama 兜底）

这是一个用于毕设展示的 RAG Demo，核心特点：

- **全在线**：Embedding / Reranker / LLM / 图片解析均可通过 **硅基流动 API** 完成；
- **Ollama 兜底**：LLM 可切换为 Ollama（OpenAI 兼容接口，仅改 URL），用于答辩说明“可本地运行、数据不外传”；本仓库不包含 Ollama 与模型文件。
- **知识组多选 + 主/次权重**：一次提问可以选择多个知识组，并对不同组设置优先级（主/次组），通过检索加权影响召回排序。
- **前端保存配置**：API Key 与模型名保存在浏览器 **LocalStorage**，后端不保存密钥。

---

## 功能

- **对话**：多轮记忆（按 `thread_id`），支持“新对话”；支持附带图片提问（视觉模型先解析图→文，再问答）。
- **文档上传**：支持一次选择**多个文件**；PDF 用 PyMuPDF 提取文本；图片用硅基视觉模型提取内容。
- **知识组**：
  - 组与文档是“业务映射”（组-文档表），一个文档可属于多个组；
  - 向量库是全局单索引，只存 `doc_id + chunk 向量`，不感知组；
  - 查询时按所选组过滤并按组 priority 加权，再进行 rerank。
- **评估**：提供 `POST /evaluate`，批量样例评估（输出 answer/context 以及 answer vs expected 的 embedding 余弦相似度）。

---

## 快速开始

```bash
cd /home/ljr/PycharmProjects/rag_system_demo
uv sync
.venv/bin/python src/main.py
```

打开：

- 主页面：`http://127.0.0.1:8000/`
- 设置页：`http://127.0.0.1:8000/settings`
- API 文档：`http://127.0.0.1:8000/docs`

---

## 关键配置（前端设置页）

建议默认使用硅基 Qwen3 系列：

- Embedding：`Qwen/Qwen3-Embedding-0.6B`
- Reranker：`Qwen/Qwen3-Reranker-0.6B`
- LLM：例如 `Pro/deepseek-ai/DeepSeek-V3.2`
- 视觉模型：例如 `Qwen/Qwen2-VL-7B-Instruct`

这些配置写入 LocalStorage，并在请求体 `settings` 中发送给后端。

---

## 文档与架构说明

- 单索引 vs 多索引对比：`docs/INDEX_DESIGN.md`
- 当前版本架构与实现细节（接口/数据结构/检索流程/记忆/评估）：`docs/ARCHITECTURE.md`
- 修改记录与错误修复日志：`docs/CHANGELOG_DEBUG.md`

---

## 常见问题排查

（如需记录某次真实故障，请写入 `docs/CHANGELOG_DEBUG.md`。）


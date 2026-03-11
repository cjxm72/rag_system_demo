import os
import sys
import uuid
from typing import Optional, List, Literal
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
project_root = os.path.dirname(current_dir)
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config_loader import load_config
from src.store import (
    list_documents,
    list_groups,
    add_document,
    add_document_placeholder,
    delete_document,
    add_group,
    update_group,
    get_group,
    delete_group,
    get_embedding_model,
    update_document,
)
from src.rag_system import ensure_index, invalidate_index
from src.agent import create_workflow
from src.vision_api import describe_image
from src.doc_parser import parse_file
from src.eval_rag import evaluate_items, EvalItem
from fastapi.responses import StreamingResponse

app = FastAPI(title="企业数据管理秘书 · RAG（全在线）")
config = load_config()

# 请求体：前端 LocalStorage 的配置
class RequestSettings(BaseModel):
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    embedding_model: Optional[str] = None
    reranker_model: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    ollama_base_url: Optional[str] = None
    vision_model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class GroupSelection(BaseModel):
    id: str
    priority: Optional[float] = 1.0


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    question: str
    group_id: str = ""  # 兼容旧字段：若 groups 为空且该字段非空，则视为单组选中
    groups: Optional[List[GroupSelection]] = None  # 新字段：支持多组 + priority
    thread_id: str = "default"
    image_base64: Optional[str] = None
    history: Optional[List[ChatHistoryItem]] = None  # 供流式接口构造历史对话
    settings: Optional[RequestSettings] = None


class QueryResponse(BaseModel):
    answer: str
    thread_id: str


# URL 归一化：确保有 http(s)://
def _normalize_base_url(url: str, default: str) -> str:
    u = (url or "").strip()
    if not u:
        return default
    if u.startswith("http://") or u.startswith("https://"):
        return u
    # 兼容用户只填了域名/host:port 的情况
    return "https://" + u.lstrip("/")


# 确保 settings 转成扁平 dict 供 agent/rag 使用
def _settings_dict(s: Optional[RequestSettings]) -> dict:
    if not s:
        return {}
    api_base = _normalize_base_url(s.api_base or "", "https://api.siliconflow.cn/v1")
    ollama_base = (s.ollama_base_url or "").strip()
    if ollama_base and not (ollama_base.startswith("http://") or ollama_base.startswith("https://")):
        # ollama 多数是本地 http
        ollama_base = "http://" + ollama_base.lstrip("/")
    elif not ollama_base:
        ollama_base = "http://localhost:11434/v1"
    return {
        "api_key": s.api_key or "",
        "api_base": api_base,
        "embedding_model": s.embedding_model or "Qwen/Qwen3-Embedding-0.6B",
        "reranker_model": s.reranker_model or "Qwen/Qwen3-Reranker-0.6B",
        "llm_provider": s.llm_provider or "siliconflow",
        "llm_model": s.llm_model or "Pro/deepseek-ai/DeepSeek-V3.2",
        "ollama_base_url": ollama_base,
        "vision_model": s.vision_model or "Qwen/Qwen2-VL-7B-Instruct",
        "temperature": s.temperature if s.temperature is not None else 0.7,
        "max_tokens": s.max_tokens if s.max_tokens is not None else 2000,
    }


workflow = create_workflow()

# 静态前端
static_dir = os.path.join(project_root, "static")
if os.path.isdir(static_dir):
    @app.get("/")
    def index():
        return FileResponse(os.path.join(static_dir, "index.html"))

    @app.get("/settings")
    def settings_page():
        p = os.path.join(static_dir, "settings.html")
        if os.path.isfile(p):
            return FileResponse(p)
        return FileResponse(os.path.join(static_dir, "index.html"))

    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.post("/query", response_model=QueryResponse)
async def post_query(req: QueryRequest):
    try:
        question = req.question.strip()
        settings = _settings_dict(req.settings)
        if req.image_base64:
            desc = describe_image(
                req.image_base64,
                api_key=settings.get("api_key") or "",
                vision_model=settings.get("vision_model") or "Qwen/Qwen2-VL-7B-Instruct",
                api_base=settings.get("api_base") or "https://api.siliconflow.cn/v1",
            )
            question = f"[图片内容]\n{desc}\n\n[用户问题]\n{question}"
        # 组选择：优先使用 new groups 字段，否则回退到单一 group_id
        groups_payload: List[dict] = []
        if req.groups:
            for g in req.groups:
                groups_payload.append({"id": g.id, "priority": g.priority or 1.0})
        elif req.group_id:
            groups_payload.append({"id": req.group_id, "priority": 1.0})

        state = {
            "question": question,
            "groups": groups_payload,
            "settings": settings,
        }
        result = workflow.invoke(state, {"configurable": {"thread_id": req.thread_id}})
        answer = result["messages"][-1].content if result.get("messages") else ""
        return QueryResponse(answer=answer, thread_id=req.thread_id)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def post_query_stream(req: QueryRequest):
    """
    流式回答（text/plain 流）：先检索，再流式输出 LLM token。
    前端用 fetch 读取 ReadableStream 即可实现“边生成边显示”。
    """
    try:
        question = req.question.strip()
        settings = _settings_dict(req.settings)

        if req.image_base64:
            desc = describe_image(
                req.image_base64,
                api_key=settings.get("api_key") or "",
                vision_model=settings.get("vision_model") or "Qwen/Qwen2-VL-7B-Instruct",
                api_base=settings.get("api_base") or "https://api.siliconflow.cn/v1",
            )
            question = f"[图片内容]\n{desc}\n\n[用户问题]\n{question}"

        groups_payload: List[dict] = []
        if req.groups:
            for g in req.groups:
                groups_payload.append({"id": g.id, "priority": g.priority or 1.0})
        elif req.group_id:
            groups_payload.append({"id": req.group_id, "priority": 1.0})

        from src.agent import initialize_llm
        from src.rag_system import query as rag_query
        from langchain_core.messages import SystemMessage, HumanMessage

        api_key = settings.get("api_key") or ""
        api_base = settings.get("api_base") or "https://api.siliconflow.cn/v1"
        context = rag_query(
            question=question,
            selected_groups=groups_payload,
            api_key=api_key,
            embedding_model=settings.get("embedding_model") or "Qwen/Qwen3-Embedding-0.6B",
            reranker_model=settings.get("reranker_model") or "Qwen/Qwen3-Reranker-0.6B",
            api_base=api_base,
            rerank_top_n=config.vector_store.rerank_top_n,
            similarity_top_k=config.vector_store.similarity_top_k * 2,
        )
        # 将前端传来的历史对话拼进提示词，增强记忆感
        history_items = req.history or []
        hist_str = ""
        for h in history_items:
            prefix = "用户" if h.role == "user" else "助手"
            hist_str += f"{prefix}：{h.content}\n"

        system_prompt = config.prompts.system
        if hist_str.strip():
            system_prompt += "\n\n--- 历史对话 ---\n" + hist_str
        system_prompt += "\n\n--- 检索到的相关文档 ---\n" + context
        system_prompt += "\n\n--- 请严格根据以上内容回答当前问题，不要编造 ---"

        llm = initialize_llm(settings)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]

        def gen():
            try:
                for chunk in llm.stream(messages):
                    text = getattr(chunk, "content", "") or ""
                    if text:
                        yield text
            except Exception as e:
                yield f"\n\n[STREAM_ERROR] {e}"

        return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def get_documents():
    return {"documents": list_documents()}


@app.post("/documents/upload")
async def upload_document(
    file: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None,
    api_key: str = Form(""),
    embedding_model: str = Form("Qwen/Qwen3-Embedding-0.6B"),
    api_base: str = Form("https://api.siliconflow.cn/v1"),
    vision_model: str = Form("Qwen/Qwen2-VL-7B-Instruct"),
):
    if not file:
        raise HTTPException(400, "缺少文件")
    upload_dir = os.path.join(project_root, "data", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    results = []
    api_base = _normalize_base_url(api_base, "https://api.siliconflow.cn/v1")
    for fup in file:
        if not fup.filename:
            continue
        ext = os.path.splitext(fup.filename)[1].lower()
        path = os.path.join(upload_dir, f"{uuid.uuid4().hex}{ext}")
        content = await fup.read()
        with open(path, "wb") as wf:
            wf.write(content)
        doc_id = add_document_placeholder(name=fup.filename, path_or_note=path)
        results.append({"id": doc_id, "name": fup.filename, "status": "queued"})

        def _parse_and_index(doc_id_inner: str, path_inner: str):
            try:
                update_document(doc_id_inner, status="parsing", progress=10, error="")
                text = parse_file(
                    path_inner,
                    api_key=api_key or None,
                    vision_model=vision_model or None,
                    api_base=api_base,
                )
                update_document(doc_id_inner, text=text, status="embedding", progress=70)
                # 单索引：解析完成后立即重建索引，保证马上可检索
                ensure_index(api_key, embedding_model, api_base, force_rebuild=True)
                update_document(doc_id_inner, status="done", progress=100)
            except Exception as e:
                update_document(doc_id_inner, status="error", progress=100, error=str(e))

        if background_tasks is not None:
            background_tasks.add_task(_parse_and_index, doc_id, path)
        else:
            _parse_and_index(doc_id, path)

    return {"uploaded": results}


@app.delete("/documents/{doc_id}")
def remove_document(doc_id: str):
    delete_document(doc_id)
    invalidate_index()
    return {"ok": True}


@app.get("/groups")
def get_groups():
    return {"groups": list_groups()}


class GroupCreate(BaseModel):
    name: str
    doc_ids: Optional[List[str]] = None


@app.post("/groups")
def create_group(body: GroupCreate):
    gid = add_group(name=body.name, doc_ids=body.doc_ids)
    return {"id": gid, "name": body.name, "doc_ids": body.doc_ids or []}


class GroupUpdate(BaseModel):
    name: Optional[str] = None
    doc_ids: Optional[List[str]] = None


@app.put("/groups/{group_id}")
def put_group(group_id: str, body: GroupUpdate):
    ok = update_group(group_id, name=body.name, doc_ids=body.doc_ids)
    if not ok:
        raise HTTPException(404, "知识组不存在")
    return {"ok": True}


@app.delete("/groups/{group_id}")
def remove_group(group_id: str):
    ok = delete_group(group_id)
    if not ok:
        raise HTTPException(404, "知识组不存在")
    return {"ok": True}


class EvalRequestItem(BaseModel):
    question: str
    expected: str
    groups: Optional[List[GroupSelection]] = None


class EvalRequest(BaseModel):
    items: List[EvalRequestItem]
    settings: RequestSettings


@app.post("/evaluate")
def evaluate(req: EvalRequest):
    try:
        settings = _settings_dict(req.settings)
        items = []
        for it in req.items:
            groups_payload = []
            if it.groups:
                for g in it.groups:
                    groups_payload.append({"id": g.id, "priority": g.priority or 1.0})
            items.append(EvalItem(question=it.question, expected=it.expected, groups=groups_payload))
        return evaluate_items(items, settings=settings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

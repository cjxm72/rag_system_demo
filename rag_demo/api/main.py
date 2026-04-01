from __future__ import annotations

import os
import uuid
from typing import List, Literal, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_demo.agents.agno_team import run_team_answer
from rag_demo.eval.eval_rag import EvalItem, evaluate_items
from rag_demo.parsing.doc_parser import parse_file
from rag_demo.parsing.vision_api import describe_image
from rag_demo.rag.rag_system import ensure_index, format_citation_footer, invalidate_index
from rag_demo.rag.rag_system import query as rag_query
from rag_demo.storage.db import get_documents_by_ids
from rag_demo.storage.store import (
    add_document_placeholder,
    add_group,
    append_chat_message,
    delete_document,
    delete_group,
    list_chat_messages,
    list_documents,
    list_groups,
    update_document,
    update_group,
)

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="企业数据管理秘书 · RAG（Chroma + SQLite + Agno Team）")


class RequestSettings(BaseModel):
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    embedding_model: Optional[str] = None
    reranker_model: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_model_medical: Optional[str] = None
    llm_model_legal: Optional[str] = None
    llm_model_coordinator: Optional[str] = None
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
    group_id: str = ""
    groups: Optional[List[GroupSelection]] = None
    thread_id: str = "default"
    image_base64: Optional[str] = None
    history: Optional[List[ChatHistoryItem]] = None
    settings: Optional[RequestSettings] = None


class SourceItem(BaseModel):
    doc_id: str
    name: str


class QueryResponse(BaseModel):
    answer: str
    thread_id: str
    sources: List[SourceItem] = Field(default_factory=list)


def _normalize_base_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    if u.startswith("http://") or u.startswith("https://"):
        return u
    return "https://" + u.lstrip("/")


def _settings_dict(s: Optional[RequestSettings], *, require_vision_model: bool = True) -> dict:
    if not s:
        raise HTTPException(status_code=400, detail="缺少 settings（前端需从 LocalStorage 传入 settings）")

    api_base = _normalize_base_url(s.api_base or "")
    ollama_base = (s.ollama_base_url or "").strip()
    if ollama_base and not (ollama_base.startswith("http://") or ollama_base.startswith("https://")):
        ollama_base = "http://" + ollama_base.lstrip("/")

    settings = {
        "api_key": s.api_key or "",
        "api_base": api_base,
        "embedding_model": (s.embedding_model or "").strip(),
        "reranker_model": (s.reranker_model or "").strip(),
        "llm_provider": (s.llm_provider or "").strip(),
        "llm_model": (s.llm_model or "").strip(),
        "llm_model_medical": s.llm_model_medical,
        "llm_model_legal": s.llm_model_legal,
        "llm_model_coordinator": s.llm_model_coordinator,
        "ollama_base_url": ollama_base,
        "vision_model": (s.vision_model or "").strip(),
        "temperature": s.temperature,
        "max_tokens": s.max_tokens,
    }
    required = [
        "api_key",
        "api_base",
        "embedding_model",
        "reranker_model",
        "llm_provider",
        "llm_model",
        "temperature",
        "max_tokens",
    ]
    if require_vision_model:
        required.append("vision_model")
    missing = [k for k in required if not settings.get(k)]
    if missing:
        raise HTTPException(status_code=400, detail=f"settings 缺少字段: {', '.join(missing)}")
    return settings


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


def _groups_payload(req: QueryRequest) -> List[dict]:
    payload: List[dict] = []
    if req.groups:
        for g in req.groups:
            payload.append({"id": g.id, "priority": g.priority or 1.0})
    elif req.group_id:
        payload.append({"id": req.group_id, "priority": 1.0})
    return payload


def _user_visible_question(req: QueryRequest, settings: dict) -> str:
    q = (req.question or "").strip()
    if req.image_base64:
        desc = describe_image(
            req.image_base64,
            api_key=settings.get("api_key") or "",
            vision_model=settings.get("vision_model") or "",
            api_base=settings.get("api_base") or "",
        )
        q = f"[图片内容]\n{desc}\n\n[用户问题]\n{q}"
    return q


def _history_lines(prior: List[dict]) -> str:
    if not prior:
        return ""
    lines: List[str] = []
    for m in prior:
        prefix = "用户" if m.get("role") == "user" else "助手"
        lines.append(f"{prefix}：{m.get('content', '')}")
    return "\n".join(lines)


def _question_for_rag(user_visible: str, prior: List[dict]) -> str:
    hb = _history_lines(prior)
    if not hb.strip():
        return user_visible
    return f"--- 历史对话 ---\n{hb}\n--- 当前问题 ---\n{user_visible}"


def _fallback_history(req: QueryRequest) -> List[dict]:
    if not req.history:
        return []
    return [{"role": h.role, "content": h.content} for h in req.history]


def _sources_from_rag(doc_ids: List[str]) -> List[SourceItem]:
    if not doc_ids:
        return []
    metas = get_documents_by_ids(doc_ids)
    out: List[SourceItem] = []
    for did in doc_ids:
        m = metas.get(did) or {}
        out.append(SourceItem(doc_id=did, name=m.get("name") or did))
    return out


@app.post("/query", response_model=QueryResponse)
async def post_query(req: QueryRequest):
    try:
        settings = _settings_dict(req.settings)
        prior = list_chat_messages(req.thread_id)
        if not prior:
            prior = _fallback_history(req)

        user_visible = _user_visible_question(req, settings)
        q_for_rag = _question_for_rag(user_visible, prior)
        append_chat_message(req.thread_id, "user", user_visible)
        groups_payload = _groups_payload(req)

        api_key = settings.get("api_key") or ""
        api_base = settings.get("api_base") or ""
        rag_res = rag_query(
            question=q_for_rag,
            selected_groups=groups_payload,
            api_key=api_key,
            embedding_model=settings.get("embedding_model") or "",
            reranker_model=settings.get("reranker_model") or "",
            api_base=api_base,
            rerank_top_n=5,
            similarity_top_k=10,
        )

        footer = format_citation_footer(rag_res.citation_doc_ids)
        answer, decision, _ = run_team_answer(
            question=q_for_rag,
            context=rag_res.context_text,
            settings=settings,
            stream=False,
            route_question=user_visible,
        )
        body = (answer or "").strip()
        body = f"【路由说明】{decision.domain}（{decision.reason}）\n\n{body}{footer}"
        append_chat_message(req.thread_id, "assistant", body.strip())

        return QueryResponse(
            answer=body.strip(),
            thread_id=req.thread_id,
            sources=_sources_from_rag(rag_res.citation_doc_ids),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def post_query_stream(req: QueryRequest):
    try:
        settings = _settings_dict(req.settings)
        prior = list_chat_messages(req.thread_id)
        if not prior:
            prior = _fallback_history(req)

        user_visible = _user_visible_question(req, settings)
        q_for_rag = _question_for_rag(user_visible, prior)
        append_chat_message(req.thread_id, "user", user_visible)
        groups_payload = _groups_payload(req)

        api_key = settings.get("api_key") or ""
        api_base = settings.get("api_base") or ""
        rag_res = rag_query(
            question=q_for_rag,
            selected_groups=groups_payload,
            api_key=api_key,
            embedding_model=settings.get("embedding_model") or "",
            reranker_model=settings.get("reranker_model") or "",
            api_base=api_base,
            rerank_top_n=5,
            similarity_top_k=10,
        )

        footer = format_citation_footer(rag_res.citation_doc_ids)
        _, decision, it = run_team_answer(
            question=q_for_rag,
            context=rag_res.context_text,
            settings=settings,
            stream=True,
            route_question=user_visible,
        )

        def gen():
            buf: List[str] = []
            yield f"【路由说明】{decision.domain}（{decision.reason}）\n\n"
            try:
                assert it is not None
                for t in it:
                    buf.append(t)
                    yield t
            except Exception as e:
                yield f"\n\n[STREAM_ERROR] {e}"
            yield footer
            full = "".join(buf) + footer
            append_chat_message(req.thread_id, "assistant", full.strip())

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
    api_key: str = Form(...),
    embedding_model: str = Form(...),
    api_base: str = Form(...),
    vision_model: str = Form(...),
):
    if not file:
        raise HTTPException(400, "缺少文件")
    upload_dir = os.path.join(project_root, "data", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    results = []
    api_base = _normalize_base_url(api_base)
    missing = []
    if not (api_key or "").strip():
        missing.append("api_key")
    if not (embedding_model or "").strip():
        missing.append("embedding_model")
    if not (api_base or "").strip():
        missing.append("api_base")
    if not (vision_model or "").strip():
        missing.append("vision_model")
    if missing:
        raise HTTPException(status_code=400, detail=f"表单缺少字段: {', '.join(missing)}")

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
        # 评估不涉及图片输入，不强制 vision_model
        settings = _settings_dict(req.settings, require_vision_model=False)
        items: List[EvalItem] = []
        for it in req.items:
            groups_payload: List[dict] = []
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

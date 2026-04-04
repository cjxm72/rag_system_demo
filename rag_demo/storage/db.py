"""
PostgreSQL 持久化（SQLModel）：文档、知识组、对话记忆。
通过环境变量 DATABASE_URL 连接，例如：
  postgresql+psycopg://user:pass@localhost:5432/ragdb
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, select
from sqlmodel import Session

from rag_demo.storage.database import create_db_and_tables, session_scope
from rag_demo.storage.models import AppMeta, ChatMessage, Document, Group, GroupMember

_initialized = False


def _pg_safe_str(value: Any) -> str:
    """PostgreSQL text/varchar 不允许 NUL，迁移或解析二进制误入库时需剔除。"""
    if value is None:
        return ""
    s = str(value)
    return s.replace("\x00", "") if "\x00" in s else s


def init_db() -> None:
    global _initialized
    if _initialized:
        return
    create_db_and_tables()
    _initialized = True


def truncate_all_tables() -> None:
    """测试/运维：清空业务表（保留表结构）。按外键依赖顺序删除。"""
    init_db()
    with session_scope() as s:
        s.exec(delete(GroupMember))
        s.exec(delete(Group))
        s.exec(delete(Document))
        s.exec(delete(ChatMessage))
        s.exec(delete(AppMeta))


# --- documents ---


def list_documents() -> List[Dict[str, Any]]:
    init_db()
    with session_scope() as s:
        rows = s.scalars(select(Document).order_by(Document.created_at, Document.id)).all()
    return [
        {
            "id": r.id,
            "name": r.name,
            "path": r.path,
            "text": r.text,
            "status": r.status,
            "progress": r.progress,
            "error": r.error,
        }
        for r in rows
    ]


def get_document(doc_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    with session_scope() as s:
        r = s.get(Document, doc_id)
    if not r:
        return None
    return {
        "id": r.id,
        "name": r.name,
        "path": r.path,
        "text": r.text,
        "status": r.status,
        "progress": r.progress,
        "error": r.error,
    }


def get_documents_by_ids(doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not doc_ids:
        return {}
    init_db()
    uniq = list(dict.fromkeys(doc_ids))
    with session_scope() as s:
        rows = s.scalars(select(Document).where(Document.id.in_(uniq))).all()
    return {
        r.id: {
            "id": r.id,
            "name": r.name,
            "path": r.path,
            "text": r.text,
            "status": r.status,
            "progress": r.progress,
            "error": r.error,
        }
        for r in rows
    }


def add_document(name: str, path_or_note: str, text: str) -> str:
    init_db()
    doc_id = str(uuid.uuid4())
    with session_scope() as s:
        s.add(
            Document(
                id=doc_id,
                name=_pg_safe_str(name),
                path=_pg_safe_str(path_or_note),
                text=_pg_safe_str(text),
                status="done",
                progress=100,
                error="",
            )
        )
    return doc_id


def add_document_placeholder(name: str, path_or_note: str) -> str:
    init_db()
    doc_id = str(uuid.uuid4())
    with session_scope() as s:
        s.add(
            Document(
                id=doc_id,
                name=_pg_safe_str(name),
                path=_pg_safe_str(path_or_note),
                text="",
                status="queued",
                progress=0,
                error="",
            )
        )
    return doc_id


def update_document(doc_id: str, **fields: Any) -> bool:
    if not fields:
        return False
    init_db()
    with session_scope() as s:
        r = s.get(Document, doc_id)
        if not r:
            return False
        for k, v in fields.items():
            if hasattr(r, k):
                if k in ("name", "path", "text", "error", "status") and isinstance(v, str):
                    v = _pg_safe_str(v)
                setattr(r, k, v)
        s.add(r)
    return True


def delete_document(doc_id: str) -> bool:
    init_db()
    with session_scope() as s:
        s.exec(delete(GroupMember).where(GroupMember.doc_id == doc_id))
        r = s.get(Document, doc_id)
        if not r:
            return False
        s.delete(r)
    return True


# --- groups ---


def list_groups() -> List[Dict[str, Any]]:
    init_db()
    with session_scope() as s:
        rows = s.scalars(select(Group).order_by(Group.name)).all()
        if not rows:
            return []
        gids = [r.id for r in rows]
        members = s.scalars(
            select(GroupMember).where(GroupMember.group_id.in_(gids)).order_by(GroupMember.group_id, GroupMember.doc_id)
        ).all()
    by_gid: Dict[str, List[str]] = {}
    for m in members:
        by_gid.setdefault(m.group_id, []).append(m.doc_id)
    return [
        {
            "id": r.id,
            "name": r.name,
            "description": getattr(r, "description", "") or "",
            "type": getattr(r, "type", "") or "",
            "doc_ids": by_gid.get(r.id, []),
        }
        for r in rows
    ]


def add_group(
    name: str,
    doc_ids: Optional[List[str]] = None,
    *,
    description: str = "",
    type: str = "",
) -> str:
    init_db()
    gid = str(uuid.uuid4())
    with session_scope() as s:
        s.add(
            Group(
                id=gid,
                name=_pg_safe_str(name),
                description=_pg_safe_str(description),
                type=_pg_safe_str(type),
            )
        )
        for did in doc_ids or []:
            s.add(GroupMember(group_id=gid, doc_id=did))
    return gid


def update_group(
    group_id: str,
    name: Optional[str] = None,
    doc_ids: Optional[List[str]] = None,
    *,
    description: Optional[str] = None,
    type: Optional[str] = None,
) -> bool:
    init_db()
    with session_scope() as s:
        g = s.get(Group, group_id)
        if not g:
            return False
        if name is not None:
            g.name = _pg_safe_str(name)
            s.add(g)
        if description is not None:
            g.description = _pg_safe_str(description)
            s.add(g)
        if type is not None:
            g.type = _pg_safe_str(type)
            s.add(g)
        if doc_ids is not None:
            s.exec(delete(GroupMember).where(GroupMember.group_id == group_id))
            for did in doc_ids:
                s.add(GroupMember(group_id=group_id, doc_id=did))
    return True


def search_groups(*, q: str = "", type: str = "", limit: int = 50) -> List[Dict[str, Any]]:
    """
    知识组检索：按 name/description 模糊匹配（ILIKE），可选 type 精确过滤。
    返回结构与 list_groups 一致（含 doc_ids）。
    """
    q = (q or "").strip()
    type = (type or "").strip()
    limit = max(1, min(200, int(limit)))

    init_db()
    with session_scope() as s:
        stmt = select(Group)
        if type:
            stmt = stmt.where(Group.type == type)
        if q:
            like = f"%{q}%"
            stmt = stmt.where((Group.name.ilike(like)) | (Group.description.ilike(like)))
        rows = s.scalars(stmt.order_by(Group.name).limit(limit)).all()
        if not rows:
            return []
        gids = [r.id for r in rows]
        members = s.scalars(
            select(GroupMember).where(GroupMember.group_id.in_(gids)).order_by(GroupMember.group_id, GroupMember.doc_id)
        ).all()
    by_gid: Dict[str, List[str]] = {}
    for m in members:
        by_gid.setdefault(m.group_id, []).append(m.doc_id)
    return [
        {
            "id": r.id,
            "name": r.name,
            "description": getattr(r, "description", "") or "",
            "type": getattr(r, "type", "") or "",
            "doc_ids": by_gid.get(r.id, []),
        }
        for r in rows
    ]


def get_group(group_id: str) -> Optional[Dict[str, Any]]:
    for g in list_groups():
        if g.get("id") == group_id:
            return g
    return None


def delete_group(group_id: str) -> bool:
    init_db()
    with session_scope() as s:
        g = s.get(Group, group_id)
        if not g:
            return False
        s.exec(delete(GroupMember).where(GroupMember.group_id == group_id))
        s.delete(g)
    return True


# --- meta ---


def get_meta(key: str, default: str = "") -> str:
    init_db()
    with session_scope() as s:
        r = s.get(AppMeta, key)
    return r.value if r else default


def set_meta(key: str, value: str) -> None:
    init_db()
    with session_scope() as s:
        s.merge(AppMeta(key=key, value=value))


def get_embedding_model() -> str:
    return get_meta("embedding_model", "")


def set_embedding_model(model: str) -> None:
    set_meta("embedding_model", model)


# --- chat memory ---


def append_chat_message(thread_id: str, role: str, content: str) -> None:
    init_db()
    with session_scope() as s:
        s.add(
            ChatMessage(
                thread_id=_pg_safe_str(thread_id),
                role=_pg_safe_str(role),
                content=_pg_safe_str(content),
                created_at=int(time.time()),
            )
        )


def list_chat_messages(thread_id: str, limit: int = 40) -> List[Dict[str, Any]]:
    init_db()
    with session_scope() as s:
        sub = (
            select(ChatMessage)
            .where(ChatMessage.thread_id == thread_id)
            .order_by(ChatMessage.id.desc())
            .limit(limit)
        )
        rows = list(reversed(s.scalars(sub).all()))
    return [{"role": r.role, "content": r.content} for r in rows]


def clear_chat_thread(thread_id: str) -> None:
    init_db()
    with session_scope() as s:
        s.exec(delete(ChatMessage).where(ChatMessage.thread_id == thread_id))

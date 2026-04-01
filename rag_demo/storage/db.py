"""
SQLite 持久化：文档、知识组、对话记忆。
路径：data/app.db（与项目根相对）。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

_lock = threading.Lock()

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
DB_PATH = os.path.join(_project_root, "data", "app.db")
_LEGACY_JSON = os.path.join(_project_root, "data", "store.json")


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_conn():
    with _lock:
        conn = _connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def init_schema() -> None:
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL DEFAULT '',
                path TEXT,
                text TEXT,
                status TEXT DEFAULT 'queued',
                progress INTEGER DEFAULT 0,
                error TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS groups (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS group_members (
                group_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                PRIMARY KEY (group_id, doc_id)
            );
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at INTEGER DEFAULT (strftime('%s','now'))
            );
            CREATE INDEX IF NOT EXISTS idx_chat_thread ON chat_messages(thread_id, id);
            CREATE TABLE IF NOT EXISTS app_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )


def _migrate_from_json_if_needed() -> None:
    if not os.path.isfile(_LEGACY_JSON):
        return
    with get_conn() as conn:
        n = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        if n > 0:
            return
    try:
        with open(_LEGACY_JSON, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return
    docs = raw.get("documents") or []
    groups = raw.get("groups") or []
    emb = raw.get("embedding_model") or ""
    with get_conn() as conn:
        for d in docs:
            conn.execute(
                """INSERT OR REPLACE INTO documents
                (id, name, path, text, status, progress, error) VALUES (?,?,?,?,?,?,?)""",
                (
                    d.get("id"),
                    d.get("name") or "",
                    d.get("path") or "",
                    d.get("text") or "",
                    d.get("status") or "done",
                    int(d.get("progress") or 0),
                    d.get("error") or "",
                ),
            )
        for g in groups:
            gid = g.get("id")
            if not gid:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO groups (id, name) VALUES (?,?)",
                (gid, g.get("name") or ""),
            )
            for did in g.get("doc_ids") or []:
                conn.execute(
                    "INSERT OR IGNORE INTO group_members (group_id, doc_id) VALUES (?,?)",
                    (gid, did),
                )
        if emb:
            conn.execute(
                "INSERT OR REPLACE INTO app_meta (key, value) VALUES (?,?)",
                ("embedding_model", emb),
            )


def init_db() -> None:
    init_schema()
    _migrate_from_json_if_needed()


# --- documents ---


def list_documents() -> List[Dict[str, Any]]:
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, name, path, text, status, progress, error FROM documents ORDER BY created_at"
        ).fetchall()
    return [dict(r) for r in rows]


def get_document(doc_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, name, path, text, status, progress, error FROM documents WHERE id=?",
            (doc_id,),
        ).fetchone()
    return dict(row) if row else None


def get_documents_by_ids(doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not doc_ids:
        return {}
    init_db()
    uniq = list(dict.fromkeys(doc_ids))
    placeholders = ",".join("?" * len(uniq))
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT id, name, path, text, status, progress, error FROM documents WHERE id IN ({placeholders})",
            uniq,
        ).fetchall()
    return {r["id"]: dict(r) for r in rows}


def add_document(name: str, path_or_note: str, text: str) -> str:
    init_db()
    doc_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO documents (id, name, path, text, status, progress, error)
            VALUES (?,?,?,?,?,?,?)""",
            (doc_id, name, path_or_note, text, "done", 100, ""),
        )
    return doc_id


def add_document_placeholder(name: str, path_or_note: str) -> str:
    init_db()
    doc_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO documents (id, name, path, text, status, progress, error)
            VALUES (?,?,?,?,?,?,?)""",
            (doc_id, name, path_or_note, "", "queued", 0, ""),
        )
    return doc_id


def update_document(doc_id: str, **fields: Any) -> bool:
    if not fields:
        return False
    init_db()
    cols = []
    vals: List[Any] = []
    for k, v in fields.items():
        cols.append(f"{k} = ?")
        vals.append(v)
    vals.append(doc_id)
    with get_conn() as conn:
        cur = conn.execute(f"UPDATE documents SET {', '.join(cols)} WHERE id = ?", vals)
        return cur.rowcount > 0


def delete_document(doc_id: str) -> bool:
    init_db()
    with get_conn() as conn:
        conn.execute("DELETE FROM group_members WHERE doc_id=?", (doc_id,))
        cur = conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        return cur.rowcount > 0


# --- groups ---


def list_groups() -> List[Dict[str, Any]]:
    init_db()
    with get_conn() as conn:
        rows = conn.execute("SELECT id, name FROM groups ORDER BY name").fetchall()
        if not rows:
            return []
        gids = [r["id"] for r in rows]
        ph = ",".join("?" * len(gids))
        members = conn.execute(
            f"SELECT group_id, doc_id FROM group_members WHERE group_id IN ({ph}) ORDER BY group_id, doc_id",
            gids,
        ).fetchall()
    by_gid: Dict[str, List[str]] = {}
    for m in members:
        by_gid.setdefault(m["group_id"], []).append(m["doc_id"])
    return [{"id": r["id"], "name": r["name"], "doc_ids": by_gid.get(r["id"], [])} for r in rows]


def add_group(name: str, doc_ids: Optional[List[str]] = None) -> str:
    init_db()
    gid = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute("INSERT INTO groups (id, name) VALUES (?,?)", (gid, name))
        for did in doc_ids or []:
            conn.execute(
                "INSERT OR IGNORE INTO group_members (group_id, doc_id) VALUES (?,?)",
                (gid, did),
            )
    return gid


def update_group(group_id: str, name: Optional[str] = None, doc_ids: Optional[List[str]] = None) -> bool:
    init_db()
    with get_conn() as conn:
        row = conn.execute("SELECT 1 FROM groups WHERE id=?", (group_id,)).fetchone()
        if not row:
            return False
        if name is not None:
            conn.execute("UPDATE groups SET name=? WHERE id=?", (name, group_id))
        if doc_ids is not None:
            conn.execute("DELETE FROM group_members WHERE group_id=?", (group_id,))
            for did in doc_ids:
                conn.execute(
                    "INSERT INTO group_members (group_id, doc_id) VALUES (?,?)",
                    (group_id, did),
                )
        return True


def get_group(group_id: str) -> Optional[Dict[str, Any]]:
    for g in list_groups():
        if g.get("id") == group_id:
            return g
    return None


def delete_group(group_id: str) -> bool:
    init_db()
    with get_conn() as conn:
        conn.execute("DELETE FROM group_members WHERE group_id=?", (group_id,))
        cur = conn.execute("DELETE FROM groups WHERE id=?", (group_id,))
        return cur.rowcount > 0


# --- meta ---


def get_meta(key: str, default: str = "") -> str:
    init_db()
    with get_conn() as conn:
        row = conn.execute("SELECT value FROM app_meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def set_meta(key: str, value: str) -> None:
    init_db()
    with get_conn() as conn:
        conn.execute("INSERT OR REPLACE INTO app_meta (key, value) VALUES (?,?)", (key, value))


def get_embedding_model() -> str:
    return get_meta("embedding_model", "")


def set_embedding_model(model: str) -> None:
    set_meta("embedding_model", model)


# --- chat memory ---


def append_chat_message(thread_id: str, role: str, content: str) -> None:
    init_db()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO chat_messages (thread_id, role, content) VALUES (?,?,?)",
            (thread_id, role, content),
        )


def list_chat_messages(thread_id: str, limit: int = 40) -> List[Dict[str, Any]]:
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT id, role, content, created_at FROM chat_messages
            WHERE thread_id=? ORDER BY id DESC LIMIT ?""",
            (thread_id, limit),
        ).fetchall()
    rows = list(reversed(rows))
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def clear_chat_thread(thread_id: str) -> None:
    init_db()
    with get_conn() as conn:
        conn.execute("DELETE FROM chat_messages WHERE thread_id=?", (thread_id,))


init_db()

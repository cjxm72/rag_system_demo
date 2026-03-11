"""
文档与知识组持久化：单文件 data/store.json。
单索引 + metadata 过滤：所有 chunk 进同一 FAISS，metadata 含 doc_id；查询时按 group 的 doc_ids 过滤。
"""
import os
import json
import uuid
from typing import List, Dict, Any, Optional

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
STORE_PATH = os.path.join(_project_root, "data", "store.json")


def _load_raw() -> dict:
    if not os.path.isfile(STORE_PATH):
        return {"documents": [], "groups": [], "embedding_model": ""}
    with open(STORE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_raw(data: dict) -> None:
    os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def list_documents() -> List[Dict[str, Any]]:
    return _load_raw().get("documents", [])


def list_groups() -> List[Dict[str, Any]]:
    return _load_raw().get("groups", [])


def get_embedding_model() -> str:
    return _load_raw().get("embedding_model", "")


def set_embedding_model(model: str) -> None:
    data = _load_raw()
    data["embedding_model"] = model
    _save_raw(data)


def add_document(name: str, path_or_note: str, text: str) -> str:
    data = _load_raw()
    doc_id = str(uuid.uuid4())
    data.setdefault("documents", []).append({
        "id": doc_id,
        "name": name,
        "path": path_or_note,
        "text": text,
        "status": "done",
        "progress": 100,
        "error": "",
    })
    _save_raw(data)
    return doc_id


def add_document_placeholder(name: str, path_or_note: str) -> str:
    """上传后先占位，后台解析时更新 text/status/progress。"""
    data = _load_raw()
    doc_id = str(uuid.uuid4())
    data.setdefault("documents", []).append({
        "id": doc_id,
        "name": name,
        "path": path_or_note,
        "text": "",
        "status": "queued",
        "progress": 0,
        "error": "",
    })
    _save_raw(data)
    return doc_id


def update_document(doc_id: str, **fields) -> bool:
    data = _load_raw()
    for d in data.get("documents", []):
        if d.get("id") == doc_id:
            for k, v in fields.items():
                d[k] = v
            _save_raw(data)
            return True
    return False


def get_document(doc_id: str) -> Dict[str, Any] | None:
    for d in list_documents():
        if d.get("id") == doc_id:
            return d
    return None


def delete_document(doc_id: str) -> bool:
    data = _load_raw()
    docs = [d for d in data.get("documents", []) if d.get("id") != doc_id]
    data["documents"] = docs
    for g in data.get("groups", []):
        g["doc_ids"] = [x for x in g.get("doc_ids", []) if x != doc_id]
    _save_raw(data)
    return True


def add_group(name: str, doc_ids: List[str] | None = None) -> str:
    data = _load_raw()
    gid = str(uuid.uuid4())
    data.setdefault("groups", []).append({
        "id": gid,
        "name": name,
        "doc_ids": list(doc_ids or []),
    })
    _save_raw(data)
    return gid


def update_group(group_id: str, name: str | None = None, doc_ids: List[str] | None = None) -> bool:
    data = _load_raw()
    for g in data.get("groups", []):
        if g.get("id") == group_id:
            if name is not None:
                g["name"] = name
            if doc_ids is not None:
                g["doc_ids"] = list(doc_ids)
            _save_raw(data)
            return True
    return False


def get_group(group_id: str) -> Dict[str, Any] | None:
    for g in list_groups():
        if g.get("id") == group_id:
            return g
    return None


def delete_group(group_id: str) -> bool:
    data = _load_raw()
    data["groups"] = [g for g in data.get("groups", []) if g.get("id") != group_id]
    _save_raw(data)
    return True

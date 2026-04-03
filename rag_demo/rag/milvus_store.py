"""
Milvus 向量存储：单 collection `rag_kb`，字段与旧 Chroma metadata 对齐。
环境变量：
  MILVUS_URI（可选，如 http://localhost:19530）
  或 MILVUS_HOST + MILVUS_PORT
  MILVUS_COLLECTION（默认 rag_kb）
  MILVUS_USER / MILVUS_PASSWORD（可选）
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

COLLECTION_DEFAULT = "rag_kb"
_connected = False


def _collection_name() -> str:
    return (os.environ.get("MILVUS_COLLECTION") or COLLECTION_DEFAULT).strip()


def _connect() -> None:
    global _connected
    if _connected:
        return
    uri = (os.environ.get("MILVUS_URI") or "").strip()
    token = (os.environ.get("MILVUS_TOKEN") or "").strip()
    user = (os.environ.get("MILVUS_USER") or "").strip()
    password = (os.environ.get("MILVUS_PASSWORD") or "").strip()
    kwargs: Dict[str, Any] = {}
    if token:
        kwargs["token"] = token
    elif user and password:
        kwargs["user"] = user
        kwargs["password"] = password
    if uri:
        connections.connect("default", uri=uri, **kwargs)
    else:
        host = (os.environ.get("MILVUS_HOST") or "localhost").strip()
        port = (os.environ.get("MILVUS_PORT") or "19530").strip()
        connections.connect("default", host=host, port=port, **kwargs)
    _connected = True


def drop_collection_if_exists(name: Optional[str] = None) -> None:
    _connect()
    n = name or _collection_name()
    if utility.has_collection(n):
        utility.drop_collection(n)


def _schema(dim: int) -> CollectionSchema:
    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=128),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65532),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    return CollectionSchema(fields, description="RAG chunks")


def open_collection_if_exists() -> Optional[Collection]:
    _connect()
    name = _collection_name()
    if not utility.has_collection(name):
        return None
    col = Collection(name)
    col.load()
    return col


def ensure_collection(dim: int) -> Collection:
    _connect()
    name = _collection_name()
    if utility.has_collection(name):
        col = Collection(name)
        col.load()
        return col
    col = Collection(name, _schema(dim))
    col.create_index(
        field_name="embedding",
        index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}},
    )
    col.load()
    return col


def insert_chunks(
    col: Collection,
    rows: Sequence[Dict[str, Any]],
) -> None:
    if not rows:
        return
    pks = [r["pk"] for r in rows]
    doc_ids = [r["doc_id"] for r in rows]
    chunk_indices = [int(r["chunk_index"]) for r in rows]
    names = [r.get("name") or "" for r in rows]
    texts = [r.get("text") or "" for r in rows]
    embeddings = [r["embedding"] for r in rows]
    col.insert([pks, doc_ids, chunk_indices, names, texts, embeddings])
    col.flush()
    col.load()


def _escape_expr_str(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def doc_id_in_expr(doc_ids: List[str]) -> str:
    if not doc_ids:
        return ""
    parts = ",".join(f'"{_escape_expr_str(d)}"' for d in doc_ids)
    return f"doc_id in [{parts}]"


def _hit_entity_dict(hit: Any) -> Dict[str, Any]:
    try:
        d = hit.to_dict()
        ent = d.get("entity") or {}
        if isinstance(ent, dict):
            return ent
    except Exception:
        pass
    ent = getattr(hit, "entity", None)
    if ent is None:
        return {}
    if isinstance(ent, dict):
        return ent
    try:
        if hasattr(ent, "to_dict"):
            raw = ent.to_dict()
            return raw if isinstance(raw, dict) else {}
    except Exception:
        pass
    out: Dict[str, Any] = {}
    for k in ("doc_id", "chunk_index", "name", "text"):
        try:
            v = ent.get(k)  # type: ignore[union-attr]
        except Exception:
            v = None
        if v is not None:
            out[k] = v
    return out


def search(
    col: Collection,
    query_vector: List[float],
    limit: int,
    expr: Optional[str] = None,
) -> List[Tuple[str, str, int, str, str, float]]:
    """
    返回 [(pk, doc_id, chunk_index, name, text, distance), ...]
    distance 为 Milvus COSINE 检索返回的 metric 值（与版本相关，后续可与 rerank 分数衔接）。
    """
    col.load()
    sp = {"metric_type": "COSINE", "params": {"nprobe": 16}}
    res = col.search(
        data=[query_vector],
        anns_field="embedding",
        param=sp,
        limit=limit,
        expr=expr if expr else None,
        output_fields=["doc_id", "chunk_index", "name", "text"],
    )
    out: List[Tuple[str, str, int, str, str, float]] = []
    for hit in res[0]:
        ent = _hit_entity_dict(hit)
        pk = str(getattr(hit, "id", "") or "")
        doc_id = str(ent.get("doc_id") or "")
        chunk_index = int(ent.get("chunk_index") or 0)
        name = str(ent.get("name") or "")
        text = str(ent.get("text") or "")
        out.append((pk, doc_id, chunk_index, name, text, float(hit.distance)))
    return out

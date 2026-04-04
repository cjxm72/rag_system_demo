"""
通过 OpenAI 兼容 POST /embeddings 对候选块做余弦相似度重排（适用于 Ollama、OpenAI 等无 /rerank 的服务）。
"""

from __future__ import annotations

import math
from typing import List

import httpx
from llama_index.core.schema import NodeWithScore


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(dot / (na * nb))


def _embed_batch(
    *,
    api_base: str,
    api_key: str,
    model: str,
    texts: List[str],
    timeout_s: float = 60.0,
    batch_size: int = 32,
) -> List[List[float]]:
    base = (api_base or "").strip().rstrip("/")
    if not base:
        raise ValueError("缺少 rerank_api_base")
    url = f"{base}/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key or 'ollama'}",
        "Content-Type": "application/json",
    }
    out: List[List[float]] = [[] for _ in texts]
    with httpx.Client(timeout=timeout_s) as client:
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            r = client.post(
                url,
                json={"model": model, "input": chunk},
                headers=headers,
            )
            r.raise_for_status()
            data = r.json()
            items = data.get("data") or []
            by_idx: dict[int, List[float]] = {}
            for j, it in enumerate(items):
                idx = it.get("index")
                if idx is None:
                    idx = j
                idx = int(idx)
                emb = it.get("embedding")
                if isinstance(emb, list):
                    by_idx[start + idx] = [float(x) for x in emb]
            for i in range(len(chunk)):
                global_i = start + i
                vec = by_idx.get(global_i)
                if vec is None:
                    raise ValueError("Embedding 响应缺少与输入对齐的向量")
                out[global_i] = vec
    return out


def postprocess_nodes_embedding_cosine(
    *,
    query: str,
    nodes: List[NodeWithScore],
    api_base: str,
    api_key: str,
    model: str,
    top_n: int,
) -> List[NodeWithScore]:
    if not nodes:
        return nodes
    texts = [n.node.get_content() or "" for n in nodes]
    all_texts = [query] + texts
    vecs = _embed_batch(api_base=api_base, api_key=api_key, model=model, texts=all_texts)
    qv = vecs[0]
    scored: List[NodeWithScore] = []
    for n, dv in zip(nodes, vecs[1:], strict=True):
        s = _cosine(qv, dv)
        scored.append(NodeWithScore(node=n.node, score=s))
    scored.sort(key=lambda x: float(x.score or 0.0), reverse=True)
    return scored[:top_n]

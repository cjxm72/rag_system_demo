"""
轻量 RAG 评估：检索 + 生成，语义相似度 proxy。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import math
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from rag_demo.core.provider_settings import finalize_settings
from rag_demo.rag.rag_system import query as rag_query
from rag_demo.rag.siliconflow_embedding import SiliconFlowEmbedding

SYSTEM_PROMPT = (
    "你是一个严谨的助理。请严格根据给定的检索上下文回答问题；若上下文不足以支撑结论，请明确说明“不确定/缺少依据”。"
)


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _initialize_llm(settings: Dict[str, Any]) -> ChatOpenAI:
    finalize_settings(settings)
    provider = (settings.get("llm_provider") or "").lower()
    model = settings.get("llm_model") or ""
    temperature = settings.get("temperature")
    max_tokens = settings.get("max_tokens")
    base = (settings.get("llm_api_base") or "").rstrip("/")
    api_key = (settings.get("llm_api_key") or "").strip()

    missing = []
    if not provider:
        missing.append("llm_provider")
    if not model:
        missing.append("llm_model")
    if temperature is None:
        missing.append("temperature")
    if max_tokens is None:
        missing.append("max_tokens")
    if not base:
        missing.append("llm_api_base")
    if provider in ("siliconflow", "openai") and not api_key:
        missing.append("llm_api_key")
    if missing:
        raise ValueError("评估 settings 缺少字段: " + ", ".join(missing))
    temperature = float(temperature)
    max_tokens = int(max_tokens)

    if provider == "ollama":
        return ChatOpenAI(
            base_url=base,
            api_key=api_key or "ollama",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return ChatOpenAI(
        base_url=base,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


@dataclass
class EvalItem:
    question: str
    expected: str
    groups: List[Dict[str, Any]]


def evaluate_items(
    items: List[EvalItem],
    settings: Dict[str, Any],
    top_k: Optional[int] = None,
    rerank_n: Optional[int] = None,
) -> Dict[str, Any]:
    finalize_settings(settings)
    embedding_model = settings.get("embedding_model") or ""
    reranker_model = settings.get("reranker_model") or ""
    emb_key = settings.get("embedding_api_key") or ""
    emb_base = settings.get("embedding_api_base") or ""

    missing = []
    if settings.get("embedding_provider") in ("siliconflow", "openai") and not emb_key:
        missing.append("embedding_api_key 或 api_key")
    if not emb_base:
        missing.append("embedding_api_base")
    if not embedding_model:
        missing.append("embedding_model")
    if not reranker_model:
        missing.append("reranker_model")
    if missing:
        raise ValueError("评估 settings 缺少字段: " + ", ".join(missing))

    top_k = int(top_k or 10)
    rerank_n = int(rerank_n or 5)

    embedder = SiliconFlowEmbedding(api_key=emb_key, api_base=emb_base, model=embedding_model)
    llm = _initialize_llm(settings)

    results = []
    scores: List[float] = []

    for it in items:
        t0 = time.time()
        last_err: Exception | None = None
        rag_res = rag_query(
            question=it.question,
            selected_groups=it.groups or [],
            settings=settings,
            similarity_top_k=top_k,
            rerank_top_n=rerank_n,
        )
        t_retrieve = time.time()
        context = rag_res.context_text
        system_prompt = (
            SYSTEM_PROMPT
            + "\n\n--- 检索到的相关文档（含 doc_id，请勿编造） ---\n"
            + context
            + "\n\n--- 请严格根据以上文档内容回答问题，不要编造 ---"
        )
        # 限流友好：LLM 429 时做指数退避重试
        answer = ""
        for attempt in range(5):
            try:
                answer = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=it.question)]).content
                last_err = None
                break
            except Exception as e:
                last_err = e
                msg = str(e)
                if "429" in msg or "rate limiting" in msg.lower() or "TPM limit" in msg:
                    time.sleep(min(2**attempt, 16))
                    continue
                raise
        if last_err is not None:
            raise RuntimeError(
                f"LLM 调用失败（可能被限流 429）。model={settings.get('llm_model')} base={settings.get('llm_api_base')}. 原始错误: {last_err}"
            ) from last_err
        t_llm = time.time()

        ea = embedder.get_text_embedding(answer or "")
        eg = embedder.get_text_embedding(it.expected or "")
        sim = _cosine(ea, eg)
        t_embed = time.time()

        results.append(
            {
                "question": it.question,
                "expected": it.expected,
                "answer": answer,
                "context": context,
                "semantic_similarity": sim,
                # 论文/调试友好明细
                "retrieval": {
                    "similarity_top_k": top_k,
                    "rerank_top_n": rerank_n,
                    "citation_doc_ids": rag_res.citation_doc_ids,
                    "chunks": [
                        {
                            "source_id": c.source_id,
                            "doc_id": c.doc_id,
                            "chunk_index": c.chunk_index,
                            "score": c.score,
                            "text": c.text,
                        }
                        for c in (rag_res.chunks or [])
                    ],
                },
                "timing_s": {
                    "retrieve": round(t_retrieve - t0, 4),
                    "llm": round(t_llm - t_retrieve, 4),
                    "embeddings_for_metrics": round(t_embed - t_llm, 4),
                    "total": round(t_embed - t0, 4),
                },
                "stats": {
                    "context_chars": len(context or ""),
                    "answer_chars": len(answer or ""),
                    "expected_chars": len((it.expected or "")),
                    "num_chunks": len(rag_res.chunks or []),
                    "num_citation_docs": len(rag_res.citation_doc_ids or []),
                },
            }
        )
        scores.append(sim)

    avg = sum(scores) / len(scores) if scores else 0.0
    return {
        "count": len(results),
        "avg_semantic_similarity": avg,
        "items": results,
    }

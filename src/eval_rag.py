"""
轻量 RAG 评估（不引入复杂外部评测框架）：
- 对每条样例：跑一次 /query 的同款流程（检索+生成）
- 输出：answer、context、以及 answer vs expected 的 embedding 余弦相似度（语义相关性 proxy）

说明：
- 这不是严格的 RAGAS，但非常适合毕设展示：可量化、可复现、实现简单、与在线 embedding 模型一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import math

from src.siliconflow_embedding import SiliconFlowEmbedding
from src.rag_system import query as rag_query
from src.agent import initialize_llm
from src.config_loader import load_config
from langchain_core.messages import SystemMessage, HumanMessage

config = load_config()


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
    api_key = settings.get("api_key") or ""
    api_base = settings.get("api_base") or "https://api.siliconflow.cn/v1"
    embedding_model = settings.get("embedding_model") or "Qwen/Qwen3-Embedding-0.6B"
    reranker_model = settings.get("reranker_model") or "Qwen/Qwen3-Reranker-0.6B"

    if not api_key:
        raise ValueError("评估需要 api_key（硅基流动）")

    top_k = int(top_k or config.vector_store.similarity_top_k * 2)
    rerank_n = int(rerank_n or config.vector_store.rerank_top_n)

    embedder = SiliconFlowEmbedding(api_key=api_key, api_base=api_base, model=embedding_model)
    llm = initialize_llm(settings)

    results = []
    scores = []

    for it in items:
        context = rag_query(
            question=it.question,
            selected_groups=it.groups or [],
            api_key=api_key,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            api_base=api_base,
            similarity_top_k=top_k,
            rerank_top_n=rerank_n,
        )
        system_prompt = (
            config.prompts.system
            + "\n\n--- 检索到的相关文档 ---\n"
            + context
            + "\n\n--- 请严格根据以上文档内容回答问题，不要编造 ---"
        )
        answer = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=it.question)]).content

        ea = embedder.get_text_embedding(answer or "")
        eg = embedder.get_text_embedding(it.expected or "")
        sim = _cosine(ea, eg)

        results.append(
            {
                "question": it.question,
                "expected": it.expected,
                "answer": answer,
                "context": context,
                "semantic_similarity": sim,
            }
        )
        scores.append(sim)

    avg = sum(scores) / len(scores) if scores else 0.0
    return {
        "count": len(results),
        "avg_semantic_similarity": avg,
        "items": results,
    }


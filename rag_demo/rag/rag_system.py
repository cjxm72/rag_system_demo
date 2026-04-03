"""
RAG：OpenAI 兼容 Embedding（硅基 / OpenAI / Ollama）+ 可选硅基 Rerank，Milvus；元数据 PostgreSQL（SQLModel）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, NodeWithScore, TextNode

from rag_demo.core.provider_settings import finalize_settings
from rag_demo.rag import milvus_store
from rag_demo.rag.siliconflow_embedding import SiliconFlowEmbedding
from rag_demo.rag.siliconflow_rerank import SiliconFlowRerank
from rag_demo.rag.types import RAGResult, RetrievedChunk
from rag_demo.storage.db import get_documents_by_ids, get_meta, init_db, list_documents, list_groups, set_embedding_model, set_meta

TEXT_SPLITTER = SentenceSplitter(chunk_size=1024, chunk_overlap=200)

_milvus_col = None


def _get_embedding_model(settings: Dict[str, Any]) -> SiliconFlowEmbedding:
    return SiliconFlowEmbedding(
        api_key=settings["embedding_api_key"],
        api_base=settings["embedding_api_base"],
        model=settings["embedding_model"],
        embed_batch_size=32,
    )


def _annotate_chunk_indices(nodes: List) -> None:
    per_doc: Dict[str, int] = {}
    for n in nodes:
        if n.metadata is None:
            n.metadata = {}
        did = str(n.metadata.get("doc_id") or "")
        per_doc[did] = per_doc.get(did, 0) + 1
        n.metadata["chunk_index"] = per_doc[did]


def _build_index(settings: Dict[str, Any]):
    global _milvus_col
    init_db()
    docs_raw = list_documents()
    if not docs_raw:
        milvus_store.drop_collection_if_exists()
        _milvus_col = None
        return None

    embed = _get_embedding_model(settings)
    embedding_model = settings["embedding_model"]
    documents: List[Document] = []
    for d in docs_raw:
        text = d.get("text") or ""
        doc_id = d.get("id") or ""
        name = d.get("name") or ""
        if not text.strip():
            continue
        documents.append(Document(text=text, metadata={"doc_id": doc_id, "name": name}, id_=doc_id))

    nodes = TEXT_SPLITTER.get_nodes_from_documents(documents)
    for n in nodes:
        if n.metadata is None:
            n.metadata = {}
        n.metadata["doc_id"] = n.metadata.get("doc_id") or getattr(n, "ref_doc_id", "") or ""
    _annotate_chunk_indices(nodes)

    texts = [n.get_content() for n in nodes]
    embeddings = embed._get_text_embeddings(texts)
    if not embeddings:
        milvus_store.drop_collection_if_exists()
        _milvus_col = None
        return None
    dim = len(embeddings[0])

    milvus_store.drop_collection_if_exists()
    col = milvus_store.ensure_collection(dim)
    rows = []
    for n, vec in zip(nodes, embeddings):
        did = str(n.metadata.get("doc_id") or "")
        cidx = int(n.metadata.get("chunk_index") or 0)
        name = str(n.metadata.get("name") or "")
        body = (n.get_content() or "")[:65530]
        pk = f"{did}:{cidx}"
        rows.append(
            {
                "pk": pk,
                "doc_id": did,
                "chunk_index": cidx,
                "name": name,
                "text": body,
                "embedding": vec,
            }
        )
    milvus_store.insert_chunks(col, rows)
    set_embedding_model(embedding_model)
    set_meta("milvus_built_embedding", embedding_model)
    _milvus_col = col
    return col


def _last_index_embedding_meta() -> str:
    return get_meta("milvus_built_embedding", "") or get_meta("chroma_built_embedding", "")


def ensure_index(settings: Dict[str, Any], force_rebuild: bool = False):
    global _milvus_col
    finalize_settings(settings)
    init_db()
    embedding_model = settings["embedding_model"]
    last = _last_index_embedding_meta()
    need_rebuild = force_rebuild or (last != embedding_model)
    if not need_rebuild:
        if _milvus_col is not None:
            return _milvus_col
        ex = milvus_store.open_collection_if_exists()
        if ex is not None:
            _milvus_col = ex
            return ex
        need_rebuild = True
    if need_rebuild:
        _milvus_col = _build_index(settings)
    return _milvus_col


def _apply_rerank(
    *,
    settings: Dict[str, Any],
    question: str,
    nodes: List[NodeWithScore],
    top_n: int,
) -> List[NodeWithScore]:
    """仅 siliconflow 走 /rerank；openai/ollama 无标准 rerank 接口时按向量分数截断。"""
    finalize_settings(settings)
    provider = (settings.get("rerank_provider") or "").lower()
    model = (settings.get("reranker_model") or "").strip()
    api_key = settings.get("rerank_api_key") or ""
    api_base = settings.get("rerank_api_base") or ""
    if provider == "siliconflow" and api_key and api_base and model:
        reranker = SiliconFlowRerank(
            api_key=api_key,
            model=model,
            api_base=api_base,
            top_n=top_n,
        )
        return reranker.postprocess_nodes(question, nodes)
    nodes.sort(key=lambda x: float(x.score or 0.0), reverse=True)
    return nodes[:top_n]


def _build_doc_weight_map(selected_groups: List[Dict], lambda_group: float) -> Dict[str, float]:
    all_groups = list_groups()
    gid_to_docs = {g.get("id"): set(g.get("doc_ids", [])) for g in all_groups}

    doc_w: Dict[str, float] = {}
    for g in selected_groups:
        gid = g.get("id")
        priority = float(g.get("priority", 1.0) or 1.0)
        doc_ids = gid_to_docs.get(gid, set())
        for d in doc_ids:
            doc_w[d] = max(doc_w.get(d, 0.0), priority)
    return {k: lambda_group * v for k, v in doc_w.items()}


def _format_context_with_sql_meta(chunks: List[RetrievedChunk]) -> str:
    doc_ids = list(dict.fromkeys(c.doc_id for c in chunks))
    metas = get_documents_by_ids(doc_ids)
    lines: List[str] = []
    for ch in chunks:
        m = metas.get(ch.doc_id) or {}
        title = m.get("name") or ch.doc_id
        lines.append(
            f"[来源{ch.source_id}] doc_id={ch.doc_id} 文档名={title} chunk={ch.chunk_index}\n{ch.text}"
        )
    return "\n\n---\n\n".join(lines)


def query(
    *,
    question: str,
    selected_groups: List[Dict],
    settings: Dict[str, Any],
    rerank_top_n: int = 5,
    similarity_top_k: int = 20,
    lambda_group: float = 1.0,
    background_weight: float = 0.0,
) -> RAGResult:
    finalize_settings(settings)
    col = ensure_index(settings)
    if col is None:
        return RAGResult(context_text="当前没有任何文档，请先上传文档并加入知识组。", chunks=[], citation_doc_ids=[])

    embed = _get_embedding_model(settings)
    qvec = embed._get_query_embedding(question)

    doc_weight_map = _build_doc_weight_map(selected_groups, lambda_group) if selected_groups else {}
    selected_doc_ids = set(doc_weight_map.keys())

    expr = None
    if selected_groups:
        expr = milvus_store.doc_id_in_expr(list(selected_doc_ids))

    hits = milvus_store.search(col, qvec, limit=similarity_top_k, expr=expr)
    if not hits:
        return RAGResult(
            context_text="未在所选知识组中检索到相关内容。" if selected_groups else "未检索到任何相关内容。",
            chunks=[],
            citation_doc_ids=[],
        )

    nodes: List[NodeWithScore] = []
    for _pk, doc_id, chunk_index, name, text, dist in hits:
        node = TextNode(
            text=text,
            metadata={"doc_id": doc_id, "chunk_index": chunk_index, "name": name},
        )
        base_score = float(dist)
        w_doc = doc_weight_map.get(doc_id, background_weight)
        score = base_score * (1.0 + w_doc)
        nodes.append(NodeWithScore(node=node, score=score))

    nodes.sort(key=lambda x: float(x.score or 0.0), reverse=True)
    nodes = nodes[:similarity_top_k]

    nodes = _apply_rerank(settings=settings, question=question, nodes=nodes, top_n=rerank_top_n)

    chunks: List[RetrievedChunk] = []
    citation_ids: List[str] = []
    for i, n in enumerate(nodes, start=1):
        did = str((n.node.metadata or {}).get("doc_id") or "")
        cidx = int((n.node.metadata or {}).get("chunk_index") or 0)
        text = n.node.get_content()
        chunks.append(
            RetrievedChunk(
                source_id=i,
                doc_id=did,
                chunk_index=cidx,
                score=float(n.score or 0.0),
                text=text,
            )
        )
        if did and did not in citation_ids:
            citation_ids.append(did)

    context_text = _format_context_with_sql_meta(chunks)
    return RAGResult(context_text=context_text, chunks=chunks, citation_doc_ids=citation_ids)


def invalidate_index() -> None:
    global _milvus_col
    _milvus_col = None
    milvus_store.drop_collection_if_exists()


def format_citation_footer(doc_ids: List[str]) -> str:
    """根据 PostgreSQL 文档表反查文件名，生成文末引用块。"""
    if not doc_ids:
        return ""
    metas = get_documents_by_ids(doc_ids)
    lines: List[str] = []
    for i, did in enumerate(doc_ids, start=1):
        m = metas.get(did) or {}
        name = m.get("name") or did
        lines.append(f"{i}. 《{name}》  doc_id={did}")
    return "\n\n---\n【引用文档】\n" + "\n".join(lines)

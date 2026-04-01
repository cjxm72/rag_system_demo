"""
全在线 RAG：硅基流动 Embedding + Reranker，Chroma 向量库，metadata 含 doc_id；
知识组信息在业务侧（SQLite group_members），支持多组 & 主/次组权重。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, NodeWithScore
from llama_index.vector_stores.chroma import ChromaVectorStore

from rag_demo.rag.siliconflow_embedding import SiliconFlowEmbedding
from rag_demo.rag.siliconflow_rerank import SiliconFlowRerank
from rag_demo.rag.types import RAGResult, RetrievedChunk
from rag_demo.storage.db import get_documents_by_ids, get_meta, init_db, list_documents, list_groups, set_embedding_model, set_meta

TEXT_SPLITTER = SentenceSplitter(chunk_size=1024, chunk_overlap=200)

_index: Optional[VectorStoreIndex] = None
_chroma_client: Optional[chromadb.PersistentClient] = None

COLLECTION_NAME = "rag_kb"


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _chroma_path() -> str:
    return os.path.join(_project_root(), "data", "chroma")


def _get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(_chroma_path(), exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=_chroma_path())
    return _chroma_client


def _get_embedding_model(api_key: str, embedding_model: str, api_base: str) -> SiliconFlowEmbedding:
    return SiliconFlowEmbedding(
        api_key=api_key,
        api_base=api_base,
        model=embedding_model,
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


def _build_index(api_key: str, embedding_model: str, api_base: str) -> Optional[VectorStoreIndex]:
    global _index
    init_db()
    docs_raw = list_documents()
    if not docs_raw:
        _index = None
        return None

    embed = _get_embedding_model(api_key, embedding_model, api_base)
    client = _get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage = StorageContext.from_defaults(vector_store=vector_store)

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

    _index = VectorStoreIndex(nodes=nodes, storage_context=storage, embed_model=embed)
    set_embedding_model(embedding_model)
    set_meta("chroma_built_embedding", embedding_model)
    return _index


def _should_rebuild(api_key: str, embedding_model: str, force_rebuild: bool) -> bool:
    if force_rebuild:
        return True
    if _index is None:
        return True
    last = get_meta("chroma_built_embedding", "")
    if last != embedding_model:
        return True
    return False


def ensure_index(api_key: str, embedding_model: str, api_base: str, force_rebuild: bool = False) -> Optional[VectorStoreIndex]:
    global _index
    if _should_rebuild(api_key, embedding_model, force_rebuild):
        _index = _build_index(api_key, embedding_model, api_base)
    return _index


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
    api_key: str,
    embedding_model: str,
    reranker_model: str,
    api_base: str,
    rerank_top_n: int = 5,
    similarity_top_k: int = 20,
    lambda_group: float = 1.0,
    background_weight: float = 0.0,
) -> RAGResult:
    index = ensure_index(api_key, embedding_model, api_base)
    if index is None:
        return RAGResult(context_text="当前没有任何文档，请先上传文档并加入知识组。", chunks=[], citation_doc_ids=[])

    doc_weight_map = _build_doc_weight_map(selected_groups, lambda_group) if selected_groups else {}
    selected_doc_ids = set(doc_weight_map.keys())

    vector_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    nodes: List[NodeWithScore] = vector_retriever.retrieve(question)
    if not nodes:
        return RAGResult(context_text="未检索到任何相关内容。", chunks=[], citation_doc_ids=[])

    if selected_groups:
        nodes = [n for n in nodes if (n.node.metadata or {}).get("doc_id") in selected_doc_ids]
        if not nodes:
            return RAGResult(context_text="未在所选知识组中检索到相关内容。", chunks=[], citation_doc_ids=[])

    for n in nodes:
        base_score = float(n.score or 0.0)
        doc_id = (n.node.metadata or {}).get("doc_id")
        w_doc = doc_weight_map.get(doc_id, background_weight)
        n.score = base_score * (1.0 + w_doc)

    nodes.sort(key=lambda x: float(x.score or 0.0), reverse=True)
    nodes = nodes[:similarity_top_k]

    reranker = SiliconFlowRerank(
        api_key=api_key,
        model=reranker_model,
        api_base=api_base,
        top_n=rerank_top_n,
    )
    nodes = reranker.postprocess_nodes(question, nodes)

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
    global _index
    _index = None


def format_citation_footer(doc_ids: List[str]) -> str:
    """根据 SQLite 文档表反查文件名，生成文末引用块。"""
    if not doc_ids:
        return ""
    metas = get_documents_by_ids(doc_ids)
    lines: List[str] = []
    for i, did in enumerate(doc_ids, start=1):
        m = metas.get(did) or {}
        name = m.get("name") or did
        lines.append(f"{i}. 《{name}》  doc_id={did}")
    return "\n\n---\n【引用文档】\n" + "\n".join(lines)

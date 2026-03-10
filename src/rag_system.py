"""
全在线 RAG：硅基流动 Embedding + Reranker，单索引，metadata 只存 doc_id，
知识组信息完全在业务侧（组-文档映射），支持多组 & 主/次组权重。
"""
from typing import List, Optional, Dict

import faiss
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, NodeWithScore
from llama_index.vector_stores.faiss import FaissVectorStore
from src.siliconflow_embedding import SiliconFlowEmbedding

from src.config_loader import load_config
from src.store import list_documents, list_groups, set_embedding_model
from src.siliconflow_rerank import SiliconFlowRerank

config = load_config()
TEXT_SPLITTER = SentenceSplitter(chunk_size=1024, chunk_overlap=200)

_index: Optional[VectorStoreIndex] = None


def _get_embedding_model(api_key: str, embedding_model: str, api_base: str) -> SiliconFlowEmbedding:
    return SiliconFlowEmbedding(
        api_key=api_key,
        api_base=api_base,
        model=embedding_model,
        embed_batch_size=32,
    )


def _build_index(api_key: str, embedding_model: str, api_base: str) -> Optional[VectorStoreIndex]:
    """从 store 中所有文档重建 FAISS 索引（所有 chunk 共用一个索引，metadata 仅存 doc_id）。"""
    global _index
    docs_raw = list_documents()
    if not docs_raw:
        _index = None
        return None

    embed = _get_embedding_model(api_key, embedding_model, api_base)
    dim = len(embed.get_text_embedding("test"))
    vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(dim))
    storage = StorageContext.from_defaults(vector_store=vector_store)

    documents: List[Document] = []
    for d in docs_raw:
        text = d.get("text") or ""
        doc_id = d.get("id") or ""
        name = d.get("name") or ""
        if not text.strip():
            continue
        documents.append(
            Document(text=text, metadata={"doc_id": doc_id, "name": name}, id_=doc_id)
        )

    nodes = TEXT_SPLITTER.get_nodes_from_documents(documents)
    for n in nodes:
        if n.metadata is None:
            n.metadata = {}
        n.metadata["doc_id"] = n.metadata.get("doc_id") or getattr(n, "ref_doc_id", "") or ""

    _index = VectorStoreIndex(nodes=nodes, storage_context=storage, embed_model=embed)
    set_embedding_model(embedding_model)
    return _index


def ensure_index(api_key: str, embedding_model: str, api_base: str, force_rebuild: bool = False) -> Optional[VectorStoreIndex]:
    """若无索引或 force_rebuild，则用当前 store 重建；返回 index 或 None。"""
    global _index
    if force_rebuild or _index is None:
        _index = _build_index(api_key, embedding_model, api_base)
    return _index


def _build_doc_weight_map(selected_groups: List[Dict], lambda_group: float) -> Dict[str, float]:
    """
    根据选中的组及其 priority 计算每个 doc_id 的权重。
    groups: [{id: group_id, priority: float}, ...]
    """
    all_groups = list_groups()
    gid_to_docs = {g.get("id"): set(g.get("doc_ids", [])) for g in all_groups}

    doc_w: Dict[str, float] = {}
    for g in selected_groups:
        gid = g.get("id")
        priority = float(g.get("priority", 1.0) or 1.0)
        doc_ids = gid_to_docs.get(gid, set())
        for d in doc_ids:
            # 取最大权重，表示“任何一个组非常重要就拉高该文档”
            doc_w[d] = max(doc_w.get(d, 0.0), priority)
    # 最终使用时会用 (1 + lambda_group * w_doc) 去放大得分
    return {k: lambda_group * v for k, v in doc_w.items()}


def query(
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
) -> str:
    """
    多组 + 主/次组加权检索：
    - selected_groups: [{id: group_id, priority: float}, ...]
    - 先全局向量召回，再按组-文档映射对每个 doc 给予权重，融合进得分后再 Rerank。
    """
    index = ensure_index(api_key, embedding_model, api_base)
    if index is None:
        return "当前没有任何文档，请先上传文档并加入知识组。"

    # 构建 doc_id -> 权重（已乘以 lambda_group）
    doc_weight_map = _build_doc_weight_map(selected_groups, lambda_group) if selected_groups else {}
    selected_doc_ids = set(doc_weight_map.keys())

    vector_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    nodes: List[NodeWithScore] = vector_retriever.retrieve(question)
    if not nodes:
        return "未检索到任何相关内容。"

    # 若用户选择了组，则：
    # 1. 可选地丢弃所有“不在任何选中组中的 doc”（严格按组过滤）
    # 2. 或给这些 doc 一个较低的背景权重（本实现默认 background_weight=0 -> 丢弃）
    if selected_groups:
        filtered: List[NodeWithScore] = []
        for n in nodes:
            doc_id = (n.node.metadata or {}).get("doc_id")
            if doc_id in selected_doc_ids:
                filtered.append(n)
        nodes = filtered
        if not nodes:
            return "未在所选知识组中检索到相关内容。"

    # 将组权重融合到得分中：score' = score * (1 + w_doc)
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
    context = "\n\n---\n\n".join(n.node.get_content() for n in nodes)
    return context


def invalidate_index() -> None:
    """删文档后调用，下次查询会按 store 重建索引。"""
    global _index
    _index = None

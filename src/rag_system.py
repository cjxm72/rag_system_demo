from typing import List, Optional

import torch
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core.postprocessor import SentenceTransformerRerank
from langchain_core.tools import BaseTool
import numpy as np
import faiss
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore

from config_loader import load_config

config = load_config()

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, keyword_retriever):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.hybrid_ratio = config.retrieval.hybrid_ratio

    def _retrieve(self, query: str) -> List[NodeWithScore]:
        if isinstance(query, QueryBundle):
            query_str = query.query_str
        else:
            query_str = str(query)

        vector_results = self.vector_retriever.retrieve(query_str)
        keyword_results = self.keyword_retriever.retrieve(query_str)

        all_results = {}
        for r in vector_results:
            all_results[r.node.node_id] = (r, self.hybrid_ratio * r.score)

        for r in keyword_results:
            if r.node.node_id in all_results:
                _, old_score = all_results[r.node.node_id]
                all_results[r.node.node_id] = (r, old_score + (1 - self.hybrid_ratio) * r.score)
            else:
                all_results[r.node.node_id] = (r, (1 - self.hybrid_ratio) * r.score)

        sorted_results = sorted(all_results.values(), key=lambda x: x[1], reverse=True)
        return [r[0] for r in sorted_results]

class RAGSystem:

    def __init__(self):
        self.embed_model = HuggingFaceEmbedding(model_name=config.models["embedding"].model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embed_model._model.to(device)

        test_embedding = self.embed_model.get_text_embedding("test")
        embed_dim = len(test_embedding)

        self.vector_store = FaissVectorStore(faiss_index = faiss.IndexFlatL2(embed_dim))
        self.storage_context = StorageContext.from_defaults(vector_store = self.vector_store)

        self.reranker = SentenceTransformerRerank(
            top_n = config.vector_store.similarity_top_k,
            model = config.retrieval.model_path
        )

        self.text_splitter = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=200,
            paragraph_separator="\n\n",
            secondary_chunking_regex="[^,.;!?\n]+[,.;!?\n]?",
        )
        self.index = None
        self.retriever = None

    def load_document(self,documents:List[Document]):
        nodes = self.text_splitter.get_nodes_from_documents(documents)
        self.index = VectorStoreIndex(nodes=nodes, storage_context= self.storage_context,embed_model=self.embed_model)


        vector_retriever = VectorIndexRetriever(
            index = self.index,
            similarity_top_k=config.vector_store.similarity_top_k * 2
        )
        keyword_retriever = self.index.as_retriever(
            retriever_mode="keyword",
            similarity_top_k=config.vector_store.similarity_top_k * 2,
            llm = None
        )

        self.retriever = HybridRetriever(vector_retriever,keyword_retriever)

    def query(self,question:str):
        if not self.index:
            raise ValueError("请先加载文档")

        nodes = self.retriever.retrieve(question)
        if not nodes:
            return "未找到相关结果。"
        # 返回最相关的一个片段
        print(f"[RAGSystem] 原始检索结果: {nodes[0].text}")
        return nodes[0].text

class RAGTool(BaseTool):

    name:str = "enterprise_rag"
    description:str = "企业数据检索工具，用于查询企业内部文档信息"
    rag_system: RAGSystem


    def _run(self, query: str) :
        """工具执行方法"""
        return self.rag_system.query(query)

























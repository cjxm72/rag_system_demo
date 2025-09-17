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

from config_loader import load_config

config = load_config()

class HybridRetriever:

    def __init__(self,vector_retriever,keyword_retriever):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.hybrid_ratio = config.retrieval.hybrid_ratio

    def retrieve(self,query:str):
        vector_results = self.vector_retriever(query)
        keyword_results = self.keyword_retriever(query)

        all_results = {}
        for r in vector_results:
            all_results[r.node.node_id] = (r,self.hybrid_ratio*r.score)

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
        self.vector_store = FaissVectorStore(faiss_index = faiss.IndexFlatL2(1536))
        self.storage_context = StorageContext.from_defaults(vector_store = self.vector_store)

        self.embed_model = HuggingFaceEmbedding(model_name=config.models["embedding"].model_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embed_model._model.to(device)  # ⚠️ 访问内部 SentenceTransformer 模型
        print(f"✅ Embedding model loaded on device: {device}")

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
            similarity_top_k=config.vector_store.similarity_top_k * 2
        )

        self.retriever = HybridRetriever(vector_retriever,keyword_retriever)

    def query(self,question:str):
        if not self.index:
            raise ValueError("请先加载文档")

        query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            node_postprocessors=[self.reranker],
            response_mode = "compact"
        )
        response = query_engine.query(question)
        return str(response)

class RAGTool(BaseTool):

    name:str = "enterprise_rag"
    description:str = "企业数据检索工具，用于查询企业内部文档信息"
    rag_system: RAGSystem


    def _run(self, query: str) :
        """工具执行方法"""
        return self.rag_system.query(query)

























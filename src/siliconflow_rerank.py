"""
硅基流动 Reranker：调用 POST /v1/rerank，并实现 LlamaIndex 的 NodePostprocessor 接口。
"""
import httpx
from typing import List, Optional
from llama_index.core.schema import NodeWithScore, TextNode


def rerank_siliconflow(
    query: str,
    documents: List[str],
    api_key: str,
    model: str = "BAAI/bge-reranker-v2-m3",
    api_base: str = "https://api.siliconflow.cn/v1",
    top_n: int = 5,
) -> List[dict]:
    """返回 [ {"index": 0, "relevance_score": 0.99}, ... ] 按分数降序"""
    url = api_base.rstrip("/") + "/rerank"
    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": top_n,
        "return_documents": False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=30.0) as client:
        r = client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
    results = data.get("results", [])
    return sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)


class SiliconFlowRerank:
    """对 NodeWithScore 列表按 query 做重排序，保留 top_n。"""

    def __init__(
        self,
        api_key: str,
        model: str = "BAAI/bge-reranker-v2-m3",
        api_base: str = "https://api.siliconflow.cn/v1",
        top_n: int = 5,
    ):
        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        self.top_n = top_n

    def postprocess_nodes(
        self,
        query: str,
        nodes: List[NodeWithScore],
    ) -> List[NodeWithScore]:
        if not nodes or not self.api_key:
            return nodes[: self.top_n]
        texts = [n.node.get_content() for n in nodes]
        try:
            results = rerank_siliconflow(
                query=query,
                documents=texts,
                api_key=self.api_key,
                model=self.model,
                api_base=self.api_base,
                top_n=self.top_n,
            )
        except Exception:
            return nodes[: self.top_n]
        index_to_score = {r["index"]: r.get("relevance_score", 0) for r in results}
        out = []
        for idx, n in enumerate(nodes):
            if idx in index_to_score:
                out.append(NodeWithScore(node=n.node, score=index_to_score[idx]))
        out.sort(key=lambda x: x.score, reverse=True)
        return out[: self.top_n]

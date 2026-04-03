"""
OpenAI 兼容 Embedding（POST /v1/embeddings）：适用于硅基流动、OpenAI、Ollama 等。
"""

from __future__ import annotations

from typing import List

import httpx
from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr


class SiliconFlowEmbedding(BaseEmbedding):
    _api_key: str = PrivateAttr()
    _api_base: str = PrivateAttr()
    _model: str = PrivateAttr()
    _timeout_s: float = PrivateAttr()

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str,
        timeout_s: float = 60.0,
        embed_batch_size: int = 32,
    ):
        super().__init__(embed_batch_size=embed_batch_size)
        self._api_key = api_key
        api_base = (api_base or "").strip()
        if api_base and not (api_base.startswith("http://") or api_base.startswith("https://")):
            api_base = "https://" + api_base.lstrip("/")
        if not api_base:
            raise ValueError("缺少 api_base（需由前端 LocalStorage 传入 settings.api_base）")
        self._api_base = api_base.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _post_embeddings(self, inputs: List[str]) -> List[List[float]]:
        url = f"{self._api_base}/embeddings"
        payload = {
            "model": self._model,
            "input": inputs,
        }
        try:
            with httpx.Client(timeout=self._timeout_s) as client:
                r = client.post(url, json=payload, headers=self._headers())
                r.raise_for_status()
                data = r.json()
        except httpx.ConnectError as e:
            raise RuntimeError(
                "连接 Embedding 服务失败（DNS/网络或 Base URL）。请检查 api_base 与代理设置。原始错误: "
                + str(e)
            ) from e

        items = data.get("data") or []
        out: List[List[float]] = []
        for it in items:
            emb = it.get("embedding")
            if not isinstance(emb, list):
                raise ValueError("Embedding 响应格式异常：未返回 float 向量列表")
            out.append([float(x) for x in emb])
        return out

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._post_embeddings([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._post_embeddings([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._post_embeddings(texts)

    async def _apost_embeddings(self, inputs: List[str]) -> List[List[float]]:
        url = f"{self._api_base}/embeddings"
        payload = {
            "model": self._model,
            "input": inputs,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                r = await client.post(url, json=payload, headers=self._headers())
                r.raise_for_status()
                data = r.json()
        except httpx.ConnectError as e:
            raise RuntimeError(
                "连接 Embedding 服务失败（DNS/网络或 Base URL）。请检查 api_base 与代理设置。原始错误: "
                + str(e)
            ) from e

        items = data.get("data") or []
        out: List[List[float]] = []
        for it in items:
            emb = it.get("embedding")
            if not isinstance(emb, list):
                raise ValueError("Embedding 响应格式异常：未返回 float 向量列表")
            out.append([float(x) for x in emb])
        return out

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return (await self._apost_embeddings([query]))[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return (await self._apost_embeddings([text]))[0]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return await self._apost_embeddings(texts)


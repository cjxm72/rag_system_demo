"""测试共享工具（供 conftest 与用例导入）。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest

from rag_demo.core.provider_settings import finalize_settings


def api_key() -> str | None:
    # 测试优先读取 OPENAI_*（便于直接 source tests/test.env 覆盖本机环境变量）
    return (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("SILICONFLOW_API_KEY")
        or os.environ.get("RAG_DEMO_API_KEY")
    )


def require_api_key() -> str:
    k = api_key()
    # 允许 tests/test.env 里放占位符；占位符视为“未配置”
    if not k or k.strip().lower() in {"xx", "your_key_here", "<your_key>"}:
        pytest.skip("未设置 OPENAI_API_KEY / SILICONFLOW_API_KEY / RAG_DEMO_API_KEY（或仍为占位符 xx），跳过集成测试")
    return k


def default_settings(overrides: dict | None = None) -> dict:
    k = require_api_key()
    base = {
        "api_key": k,
        "api_base": os.environ.get("RAG_TEST_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or "",
        "embedding_provider": "siliconflow",
        "rerank_provider": "siliconflow",
        "llm_provider": "siliconflow",
        "vision_provider": "siliconflow",
        "embedding_model": os.environ.get("RAG_TEST_EMBEDDING")
        or os.environ.get("EMBEDDING_MODEL")
        or "",
        "reranker_model": os.environ.get("RAG_TEST_RERANKER")
        or os.environ.get("RERANKER_MODEL")
        or "",
        "llm_model": os.environ.get("RAG_TEST_LLM")
        or os.environ.get("OPENAI_MODEL")
        or "",
        "vision_model": os.environ.get("RAG_TEST_VISION")
        or os.environ.get("OCR_MODEL")
        or "",
        "temperature": 0.3,
        "max_tokens": 1500,
    }
    if overrides:
        base.update({x: y for x, y in overrides.items() if y is not None})
    finalize_settings(base)
    missing = [
        k
        for k in (
            "embedding_api_base",
            "embedding_model",
            "reranker_model",
            "llm_model",
            "llm_api_base",
            "vision_model",
            "vision_api_base",
        )
        if not base.get(k)
    ]
    if missing:
        pytest.skip(f"测试配置不完整，缺少: {', '.join(missing)}。请 source tests/test.env 或导出对应环境变量")
    return base


def print_sse_answer_preview(test_label: str, data: Dict[str, Any], max_chars: int = 2500) -> None:
    """设置环境变量 RAG_TEST_PRINT_ANSWERS=1 且 pytest 加 -s 时，在终端打印对话/回答摘要。"""
    if not (os.environ.get("RAG_TEST_PRINT_ANSWERS") or "").strip():
        return
    ans = (data.get("answer") or "").strip()
    src = data.get("sources") or []
    bar = "=" * 72
    print(f"\n{bar}\n[RAG_TEST_PRINT_ANSWERS] {test_label}\n{bar}", flush=True)
    print(ans[:max_chars] + ("…[截断]" if len(ans) > max_chars else ""), flush=True)
    if src:
        print(f"[sources] {len(src)} 条: {src[:5]}{'…' if len(src) > 5 else ''}", flush=True)
    print(bar + "\n", flush=True)


def consume_sse_query(response: Any) -> Dict[str, Any]:
    """解析 POST /query 返回的 text/event-stream 正文，聚合 answer 文本与 sources。"""
    text = response.text or ""
    answer_parts: list[str] = []
    sources: list[dict] = []
    route_prefix = ""
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        raw = line[6:].strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        t = obj.get("type")
        if t == "route":
            dom = obj.get("domain") or ""
            reason = obj.get("reason") or ""
            route_prefix = f"【路由说明】{dom}（{reason}）\n\n"
        elif t == "chunk":
            answer_parts.append(obj.get("text") or "")
        elif t == "sources":
            sources = list(obj.get("items") or [])
    return {"answer": route_prefix + "".join(answer_parts), "sources": sources}


def make_pdf_with_pymupdf(path: Path, text: str) -> None:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), text[:8000], fontsize=11)
    doc.save(str(path))
    doc.close()

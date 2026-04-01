"""测试共享工具（供 conftest 与用例导入）。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


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
        "embedding_model": os.environ.get("RAG_TEST_EMBEDDING")
        or os.environ.get("EMBEDDING_MODEL")
        or "",
        "reranker_model": os.environ.get("RAG_TEST_RERANKER")
        or os.environ.get("RERANKER_MODEL")
        or "",
        "llm_provider": "siliconflow",
        "llm_model": os.environ.get("RAG_TEST_LLM")
        or os.environ.get("OPENAI_MODEL")
        or "",
        "vision_model": os.environ.get("RAG_TEST_VISION")
        or os.environ.get("OCR_MODEL")
        or "",
        "temperature": 0.3,
        "max_tokens": 1500,
    }
    missing = [k for k in ("api_base", "embedding_model", "reranker_model", "llm_model", "vision_model") if not base.get(k)]
    if missing:
        pytest.skip(f"测试配置不完整，缺少: {', '.join(missing)}。请 source tests/test.env 或导出对应环境变量")
    if overrides:
        base.update({x: y for x, y in overrides.items() if y is not None})
    return base


def make_pdf_with_pymupdf(path: Path, text: str) -> None:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), text[:8000], fontsize=11)
    doc.save(str(path))
    doc.close()

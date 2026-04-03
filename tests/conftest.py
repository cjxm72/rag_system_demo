"""
测试隔离：清空 PostgreSQL 业务表、重建 Milvus collection、独立上传目录。
需配置 DATABASE_URL（PostgreSQL）与 MILVUS_URI 或 MILVUS_HOST（Milvus）。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: 需要 PostgreSQL、Milvus、网络与硅基 API Key")
    config.addinivalue_line("markers", "slow: 较慢（含 LLM 调用）")


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if not k:
            continue
        os.environ.setdefault(k, v)


def pytest_sessionstart(session):
    test_env = Path(__file__).resolve().parent / "test.env"
    _load_env_file(test_env)

    root = Path(__file__).resolve().parent
    print("\n[RAG 测试] 用例与数据文件对照（对话类用例会读 tests/text/*.txt）：")
    print("  tests/test_e2e_rag.py  — 端到端：上传、知识组、SSE /query、路由与引用")
    print("  tests/test_rag_evaluate.py — POST /evaluate")
    for p in sorted((root / "text").glob("*.txt")):
        print(f"  数据: {p.name}")
    print(
        "  终端打印 SSE 聚合正文: 设置 RAG_TEST_PRINT_ANSWERS=1 再 pytest -s\n",
        flush=True,
    )


@pytest.fixture
def tests_text_dir() -> Path:
    return Path(__file__).resolve().parent / "text"


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    if not (os.environ.get("DATABASE_URL") or "").strip():
        pytest.skip("需要 DATABASE_URL（PostgreSQL），见 tests/test.env.example")
    if not (os.environ.get("MILVUS_URI") or os.environ.get("MILVUS_HOST") or "").strip():
        pytest.skip("需要 MILVUS_URI 或 MILVUS_HOST，见 tests/test.env.example")

    import rag_demo.api.main as main_mod
    import rag_demo.rag.rag_system as rs
    import rag_demo.storage.db as db

    db.truncate_all_tables()
    rs.invalidate_index()

    proj = tmp_path / "project_root"
    (proj / "data" / "uploads").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main_mod, "project_root", str(proj))

    yield

    db.truncate_all_tables()
    rs.invalidate_index()


@pytest.fixture
def bare_client():
    """仅需 PostgreSQL，用于轻量健康检查（不清理 Milvus）。"""
    if not (os.environ.get("DATABASE_URL") or "").strip():
        pytest.skip("需要 DATABASE_URL（PostgreSQL）")
    from rag_demo.api.main import app
    from starlette.testclient import TestClient

    return TestClient(app)


@pytest.fixture
def client(isolated_env):
    from rag_demo.api.main import app
    from starlette.testclient import TestClient

    return TestClient(app)

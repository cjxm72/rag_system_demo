"""
测试隔离：独立 SQLite / Chroma / 上传目录，避免污染开发用 data/。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: 需要网络与硅基 API Key")
    config.addinivalue_line("markers", "slow: 较慢（含 LLM 调用）")


def _load_env_file(path: Path) -> None:
    """
    轻量 .env 加载器（避免额外依赖）。

    - 仅处理 KEY=VALUE
    - 支持空行与 # 注释
    - 若环境变量已存在，则不覆盖（便于 CI / 手动 export）
    """
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
    # 自动读取 tests/test.env，让用户不需要手动 source/export
    test_env = Path(__file__).resolve().parent / "test.env"
    _load_env_file(test_env)


@pytest.fixture
def tests_text_dir() -> Path:
    return Path(__file__).resolve().parent / "text"


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    import rag_demo.api.main as main_mod
    import rag_demo.rag.rag_system as rs
    import rag_demo.storage.db as db

    data_root = tmp_path / "_data"
    data_root.mkdir(parents=True, exist_ok=True)
    db_path = data_root / "app.db"
    monkeypatch.setattr(db, "DB_PATH", str(db_path))
    monkeypatch.setattr(db, "_LEGACY_JSON", str(tmp_path / "__no_legacy__.json"))

    chroma_dir = data_root / "chroma"
    monkeypatch.setattr(rs, "_chroma_path", lambda: str(chroma_dir))
    rs._chroma_client = None
    rs._index = None

    proj = tmp_path / "project_root"
    (proj / "data" / "uploads").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main_mod, "project_root", str(proj))

    db.init_db()
    yield
    rs.invalidate_index()
    rs._chroma_client = None
    rs._index = None


@pytest.fixture
def client(isolated_env):
    from rag_demo.api.main import app
    from starlette.testclient import TestClient

    return TestClient(app)

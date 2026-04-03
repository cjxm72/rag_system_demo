"""
PostgreSQL 连接（DATABASE_URL 或 .env 中的 POSTGRES_* 默认值）。
表建在 schema public（连接参数显式设置 search_path）。
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional
from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import Session, SQLModel

_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        p = _project_root() / ".env"
        if p.is_file():
            load_dotenv(p)
    except ImportError:
        pass


def get_database_url() -> str:
    _load_dotenv()
    url = (os.environ.get("DATABASE_URL") or "").strip()
    if url:
        return url
    user = (os.environ.get("POSTGRES_USER") or "postgres").strip()
    password = (os.environ.get("POSTGRES_PASSWORD") or "localhost").strip()
    host = (os.environ.get("POSTGRES_HOST") or "127.0.0.1").strip()
    port = (os.environ.get("POSTGRES_PORT") or "5432").strip()
    db = (os.environ.get("POSTGRES_DB") or "postgres").strip()
    return (
        f"postgresql+psycopg://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/{db}"
    )


def get_engine() -> Engine:
    global _engine, _SessionLocal
    if _engine is None:
        _engine = create_engine(
            get_database_url(),
            pool_pre_ping=True,
            connect_args={"options": "-c search_path=public"},
        )
        _SessionLocal = sessionmaker(bind=_engine, class_=Session, expire_on_commit=False)
    return _engine


def create_db_and_tables() -> None:
    import rag_demo.storage.models  # noqa: F401 — 注册 SQLModel 表到 metadata

    SQLModel.metadata.create_all(get_engine())


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    get_engine()
    assert _SessionLocal is not None
    session: Session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

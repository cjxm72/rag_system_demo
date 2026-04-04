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

        root = _project_root()
        p_env = root / ".env"
        p_example = root / ".env.example"
        if p_env.is_file():
            load_dotenv(p_env)
        elif p_example.is_file():
            # 允许在未生成 .env 时也能按示例默认值启动
            load_dotenv(p_example)
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

    # 轻量“向前兼容迁移”：开发环境常直接 create_all，但不会自动加列。
    # 这里仅对 groups 增加字段（description/type）做安全的 ADD COLUMN IF NOT EXISTS。
    with get_engine().begin() as conn:
        conn.exec_driver_sql(
            "ALTER TABLE IF EXISTS public.groups "
            "ADD COLUMN IF NOT EXISTS description text NOT NULL DEFAULT ''"
        )
        conn.exec_driver_sql(
            "ALTER TABLE IF EXISTS public.groups "
            "ADD COLUMN IF NOT EXISTS type varchar(32) NOT NULL DEFAULT ''"
        )
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_groups_type ON public.groups (type)")
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_groups_type_name ON public.groups (type, name)")


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

"""
SQLModel 表定义（与旧 SQLite 表结构语义一致，适配 PostgreSQL）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Text, func
from sqlmodel import Field, SQLModel


class Document(SQLModel, table=True):
    __tablename__ = "documents"

    id: str = Field(primary_key=True, max_length=64)
    name: str = Field(default="")
    path: Optional[str] = Field(default=None)
    text: str = Field(default="", sa_column=Column(Text, nullable=False))
    status: str = Field(default="queued")
    progress: int = Field(default=0)
    error: str = Field(default="")
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True, server_default=func.now()),
    )


class Group(SQLModel, table=True):
    __tablename__ = "groups"

    id: str = Field(primary_key=True, max_length=64)
    name: str = Field(default="")


class GroupMember(SQLModel, table=True):
    __tablename__ = "group_members"

    group_id: str = Field(primary_key=True, max_length=64, foreign_key="groups.id")
    doc_id: str = Field(primary_key=True, max_length=64, foreign_key="documents.id")


class ChatMessage(SQLModel, table=True):
    __tablename__ = "chat_messages"

    id: Optional[int] = Field(default=None, primary_key=True)
    thread_id: str = Field(index=True, max_length=256)
    role: str = Field(max_length=32)
    content: str = Field(sa_column=Column(Text, nullable=False))
    # 与旧 SQLite 一致：存 Unix 秒（应用层写入）
    created_at: Optional[int] = Field(default=None)


class AppMeta(SQLModel, table=True):
    __tablename__ = "app_meta"

    key: str = Field(primary_key=True, max_length=128)
    value: str = Field(sa_column=Column(Text, nullable=False))

"""
文档与知识组：PostgreSQL 持久化（SQLModel，见 `rag_demo.storage.db`）。
"""

from rag_demo.storage.db import (  # noqa: F401
    add_document,
    add_document_placeholder,
    add_group,
    append_chat_message,
    clear_chat_thread,
    delete_document,
    delete_group,
    get_document,
    get_documents_by_ids,
    get_embedding_model,
    get_group,
    list_chat_messages,
    list_documents,
    list_groups,
    set_embedding_model,
    update_document,
    update_group,
)

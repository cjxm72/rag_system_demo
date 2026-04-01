from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class RetrievedChunk:
    """单条检索片段（用于引用与调试）。"""

    source_id: int
    doc_id: str
    chunk_index: int
    score: float
    text: str


@dataclass
class RAGResult:
    """检索结果：带编号上下文 + 片段元数据。"""

    context_text: str
    chunks: List[RetrievedChunk] = field(default_factory=list)
    citation_doc_ids: List[str] = field(default_factory=list)

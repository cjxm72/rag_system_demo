"""SSE：text/event-stream，每行一条 data: JSON。"""

from __future__ import annotations

import json
from typing import Any, Dict


def sse_encode(obj: Dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

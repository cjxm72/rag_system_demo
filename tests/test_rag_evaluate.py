"""RAG 评估接口 /evaluate 集成测试。"""

from __future__ import annotations

import time

import pytest

from tests.helpers import default_settings, require_api_key


def _wait_document(client, doc_id: str, timeout: float = 180.0) -> None:
    import time as t

    deadline = t.time() + timeout
    while t.time() < deadline:
        r = client.get("/documents")
        for d in r.json().get("documents", []):
            if d.get("id") == doc_id:
                if d.get("status") == "done":
                    return
                if d.get("status") == "error":
                    pytest.fail(d.get("error"))
        time.sleep(0.4)
    pytest.fail("timeout")


@pytest.mark.integration
@pytest.mark.slow
def test_evaluate_endpoint_returns_metrics(client, tests_text_dir):
    """上传文档后，对 /evaluate 返回的 semantic_similarity 与条目结构做断言。"""
    require_api_key()
    s = default_settings()

    raw = (tests_text_dir / "legal_sample.txt").read_bytes()
    up = client.post(
        "/documents/upload",
        files=[("file", ("legal_sample.txt", raw, "application/octet-stream"))],
        data={
            "api_key": s["api_key"],
            "embedding_model": s["embedding_model"],
            "api_base": s["api_base"],
            "vision_model": s["vision_model"],
        },
    )
    assert up.status_code == 200
    doc_id = up.json()["uploaded"][0]["id"]
    _wait_document(client, doc_id)

    gr = client.post("/groups", json={"name": "评估组", "doc_ids": [doc_id]})
    assert gr.status_code == 200
    gid = gr.json()["id"]

    payload = {
        "items": [
            {
                "question": "守约方在对方违约时可以主张什么权利？",
                "expected": "守约方有权要求赔偿损失并可向法院起诉。",
                "groups": [{"id": gid, "priority": 1.0}],
            }
        ],
        "settings": {
            "api_key": s["api_key"],
            "api_base": s["api_base"],
            "embedding_model": s["embedding_model"],
            "reranker_model": s["reranker_model"],
            "llm_provider": "siliconflow",
            "llm_model": s["llm_model"],
            "temperature": 0.2,
            "max_tokens": 800,
        },
    }
    r = client.post("/evaluate", json=payload)
    # 评估会触发 LLM + embedding，可能遇到 429 限流；此处提供更详细的失败信息
    assert r.status_code == 200, (
        f"status={r.status_code}\n"
        f"response={r.text}\n"
        f"llm_model={payload['settings'].get('llm_model')}\n"
        f"embedding_model={payload['settings'].get('embedding_model')}\n"
        f"reranker_model={payload['settings'].get('reranker_model')}\n"
        f"api_base={payload['settings'].get('api_base')}\n"
    )
    data = r.json()
    assert data.get("count") == 1
    assert "avg_semantic_similarity" in data
    assert isinstance(data["avg_semantic_similarity"], float)
    item = data["items"][0]
    assert "answer" in item and "context" in item
    assert "semantic_similarity" in item
    assert len(item.get("context") or "") > 20

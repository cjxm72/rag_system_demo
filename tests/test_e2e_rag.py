"""
端到端集成测试：入库 → 解析 → 建索引 → 知识组 → 问答（文档/法律/医疗/综合）→ 路由与引用。

运行前请设置：
  export DATABASE_URL=postgresql+psycopg://...
  export MILVUS_URI=http://127.0.0.1:19530
  export SILICONFLOW_API_KEY=sk-xxx

可选环境变量：
  RAG_TEST_LLM / RAG_TEST_EMBEDDING / RAG_TEST_RERANKER / RAG_TEST_API_BASE
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tests.helpers import (
    consume_sse_query,
    default_settings,
    make_pdf_with_pymupdf,
    print_sse_answer_preview,
    require_api_key,
)


def _wait_document(client, doc_id: str, timeout: float = 180.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get("/documents")
        assert r.status_code == 200
        for d in r.json().get("documents", []):
            if d.get("id") == doc_id:
                if d.get("status") == "done":
                    return d
                if d.get("status") == "error":
                    pytest.fail(f"文档解析失败: {d.get('error')}")
        time.sleep(0.4)
    pytest.fail(f"等待文档 {doc_id} 超时")


def _upload(client, path: Path, settings: dict) -> str:
    raw = path.read_bytes()
    data = {
        "api_key": settings["api_key"],
        "embedding_model": settings["embedding_model"],
        "api_base": settings.get("api_base") or settings.get("embedding_api_base") or "",
        "vision_model": settings.get("vision_model", "Qwen/Qwen2-VL-7B-Instruct"),
        "embedding_provider": settings.get("embedding_provider", "siliconflow"),
        "vision_provider": settings.get("vision_provider", "siliconflow"),
    }
    if settings.get("ollama_base_url"):
        data["ollama_base_url"] = settings["ollama_base_url"]
    r = client.post(
        "/documents/upload",
        files=[("file", (path.name, raw, "application/octet-stream"))],
        data=data,
    )
    assert r.status_code == 200, r.text
    uploaded = r.json().get("uploaded") or []
    assert len(uploaded) == 1
    return uploaded[0]["id"]


def _create_group_with_docs(client, name: str, doc_ids: list[str]) -> str:
    r = client.post("/groups", json={"name": name, "doc_ids": doc_ids})
    assert r.status_code == 200
    return r.json()["id"]


@pytest.mark.integration
@pytest.mark.slow
def test_ingest_txt_parse_and_index(client, tests_text_dir: Path):
    """纯文本入库：解析完成 + 可检索到片段（不调用 LLM，仅 RAG 检索）。"""
    require_api_key()
    s = default_settings()
    doc_id = _upload(client, tests_text_dir / "legal_sample.txt", s)
    _wait_document(client, doc_id)

    from rag_demo.rag.rag_system import query as rag_query

    r = rag_query(
        question="违约后守约方有什么权利？",
        selected_groups=[],
        settings=s,
        similarity_top_k=10,
        rerank_top_n=5,
    )
    assert "赔偿" in r.context_text or "民法典" in r.context_text
    assert len(r.citation_doc_ids) >= 1


@pytest.mark.integration
@pytest.mark.slow
def test_pdf_upload_parse(client, tmp_path: Path):
    """PDF 入库：PyMuPDF 生成样例 PDF，经解析后进入索引。"""
    require_api_key()
    s = default_settings()
    pdf_path = tmp_path / "sample_rag.pdf"
    make_pdf_with_pymupdf(
        pdf_path,
        "本 PDF 用于 RAG 测试。关键词：合同争议、诉讼、仲裁条款。",
    )
    doc_id = _upload(client, pdf_path, s)
    _wait_document(client, doc_id)
    r = client.get("/documents")
    row = next(x for x in r.json()["documents"] if x["id"] == doc_id)
    assert row["status"] == "done"
    assert len((row.get("text") or "")) > 10


@pytest.mark.integration
@pytest.mark.slow
def test_query_document_citation_and_sources(client, tests_text_dir: Path):
    """问答：文档相关问题，检查【引用文档】与 sources。"""
    require_api_key()
    s = default_settings()
    doc_id = _upload(client, tests_text_dir / "legal_sample.txt", s)
    _wait_document(client, doc_id)
    gid = _create_group_with_docs(client, "法律组", [doc_id])

    body = {
        "question": "根据上传文档，守约方在对方违约时可以主张什么？",
        "groups": [{"id": gid, "priority": 1.0}],
        "thread_id": "test-thread-citation",
        "settings": default_settings(),
    }
    r = client.post("/query", json=body)
    assert r.status_code == 200, r.text
    data = consume_sse_query(r)
    print_sse_answer_preview("test_query_document_citation_and_sources", data)
    assert "【路由说明】" in data["answer"]
    assert "【引用文档】" in data["answer"]
    assert len(data.get("sources") or []) >= 1


@pytest.mark.integration
@pytest.mark.slow
def test_routing_medical_legal_mixed(client, tests_text_dir: Path):
    """路由：医疗 / 法律 / 综合问题，检查【路由说明】中的领域标记。"""
    require_api_key()
    # 路由测试会触发多次 LLM 调用，降低 max_tokens，并限制单次请求超时，避免卡死
    s = {
        **default_settings(),
        "max_tokens": 350,
        "temperature": 0.2,
        "request_timeout_s": 90,
        "request_retries": 2,
    }
    ids = []
    for name in ("medical_sample.txt", "legal_sample.txt", "mixed_sample.txt"):
        did = _upload(client, tests_text_dir / name, s)
        _wait_document(client, did)
        ids.append(did)
    gid = _create_group_with_docs(client, "全量组", ids)

    base = {
        "groups": [{"id": gid, "priority": 1.0}],
        "thread_id": "test-thread-route",
        "settings": s,
    }

    r1 = client.post(
        "/query",
        json={**base, "question": "高血压患者日常要注意什么？常见症状有哪些？"},
    )
    assert r1.status_code == 200, r1.text
    d1 = consume_sse_query(r1)
    print_sse_answer_preview("test_routing_medical (医疗)", d1)
    assert "【路由说明】" in d1["answer"]
    assert "medical" in d1["answer"]

    r2 = client.post(
        "/query",
        json={**base, "question": "合同违约时向法院起诉前要满足什么程序？"},
    )
    assert r2.status_code == 200, r2.text
    d2 = consume_sse_query(r2)
    print_sse_answer_preview("test_routing_medical_legal_mixed (法律)", d2)
    assert "legal" in d2["answer"]

    r3 = client.post(
        "/query",
        json={**base, "question": "医院手术出现纠纷，合同里的免责条款还有效吗？能怎么索赔？"},
    )
    assert r3.status_code == 200, r3.text
    d3 = consume_sse_query(r3)
    print_sse_answer_preview("test_routing_medical_legal_mixed (综合)", d3)
    assert "mixed" in d3["answer"]


@pytest.mark.integration
@pytest.mark.slow
def test_per_model_settings_medical_legal_members(client, tests_text_dir: Path):
    """分别为医疗/法律指定模型（可与默认相同），仍能完成路由与带引用的回答。"""
    require_api_key()
    s = {**default_settings(), "max_tokens": 700, "temperature": 0.2}
    did = _upload(client, tests_text_dir / "mixed_sample.txt", s)
    _wait_document(client, did)
    gid = _create_group_with_docs(client, "混合组", [did])

    llm = s["llm_model"]
    body = {
        "question": "本案涉及医疗服务合同与损害赔偿，应如何分析？",
        "groups": [{"id": gid, "priority": 1.0}],
        "thread_id": "test-thread-models",
        "settings": {
            **s,
            "llm_model_medical": llm,
            "llm_model_legal": llm,
        },
    }
    r = client.post("/query", json=body)
    assert r.status_code == 200, r.text
    d = consume_sse_query(r)
    print_sse_answer_preview("test_per_model_settings_medical_legal_members", d)
    assert "【路由说明】" in d["answer"]
    assert "【引用文档】" in d["answer"]


@pytest.mark.integration
def test_query_sse_contains_citation_block(client, tests_text_dir: Path):
    """SSE /query：聚合后文本含【路由说明】与【引用文档】。"""
    require_api_key()
    s = {**default_settings(), "max_tokens": 700, "temperature": 0.2}
    did = _upload(client, tests_text_dir / "legal_sample.txt", s)
    _wait_document(client, did)
    gid = _create_group_with_docs(client, "流式组", [did])

    resp = client.post(
        "/query",
        json={
            "question": "摘要一下文档里关于违约与赔偿的条款。",
            "groups": [{"id": gid, "priority": 1.0}],
            "thread_id": "test-stream",
            "settings": s,
        },
    )
    assert resp.status_code == 200
    d = consume_sse_query(resp)
    print_sse_answer_preview("test_query_sse_contains_citation_block", d)
    assert "【路由说明】" in d["answer"]
    assert "【引用文档】" in d["answer"]


def test_health_no_key(bare_client):
    """无 API Key 时仍可访问文档列表（可能为空）。"""
    r = bare_client.get("/documents")
    assert r.status_code == 200

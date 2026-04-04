"""
将前端/请求中的「提供商 + 可选专用 Base/Key」解析为统一的 OpenAI 兼容 endpoint。
支持：siliconflow / openai / ollama（Embedding、LLM、Vision 均为 OpenAI 兼容 HTTP；Rerank 硅基走 /rerank，OpenAI/Ollama 走 /embeddings 余弦重排，见 rag_system）。
"""

from __future__ import annotations

from typing import Any, Dict, MutableMapping, Optional

OLLAMA_DEFAULT_BASE = "http://127.0.0.1:11434/v1"

_VALID = frozenset({"siliconflow", "openai", "ollama"})


def normalize_provider(name: Optional[str], default: str = "siliconflow") -> str:
    p = (name or default).lower().strip()
    if p not in _VALID:
        raise ValueError(f"不支持的提供商: {name!r}（允许 siliconflow / openai / ollama）")
    return p


def normalize_http_base(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    if not (u.startswith("http://") or u.startswith("https://")):
        u = "https://" + u.lstrip("/")
    return u.rstrip("/")


def default_base_for(provider: str, ollama_base: str) -> str:
    if provider == "siliconflow":
        return "https://api.siliconflow.cn/v1"
    if provider == "openai":
        return "https://api.openai.com/v1"
    ob = normalize_http_base(ollama_base) or normalize_http_base(OLLAMA_DEFAULT_BASE)
    return ob


def resolve_api_key(provider: str, specific: str, fallback: str) -> str:
    if provider == "ollama":
        return (specific or fallback or "ollama").strip() or "ollama"
    return (specific or fallback or "").strip()


def finalize_settings(
    d: MutableMapping[str, Any],
    *,
    include_rerank: bool = True,
    include_llm: bool = True,
) -> Dict[str, Any]:
    """
    写入派生字段：embedding_api_base / embedding_api_key / rerank_* / llm_* / vision_*。
    仅填写 api_base（硅基）的场景在 provider 为 siliconflow 时作为默认 Base。
    include_rerank / include_llm 为 False 时跳过（供仅做文档入库的上传接口使用）。
    """
    ollama_raw = (d.get("ollama_base_url") or OLLAMA_DEFAULT_BASE).strip()
    if ollama_raw and not (ollama_raw.startswith("http://") or ollama_raw.startswith("https://")):
        ollama_raw = "http://" + ollama_raw.lstrip("/")
    ollama_base = normalize_http_base(ollama_raw) or normalize_http_base(OLLAMA_DEFAULT_BASE)

    default_key = (d.get("api_key") or "").strip()
    legacy_base = normalize_http_base(d.get("api_base") or "")

    emb_p = normalize_provider(d.get("embedding_provider"), "siliconflow")
    d["embedding_provider"] = emb_p
    emb_base = normalize_http_base(d.get("embedding_api_base") or "")
    if not emb_base:
        emb_base = legacy_base if emb_p == "siliconflow" and legacy_base else default_base_for(emb_p, ollama_base)
    d["embedding_api_base"] = emb_base
    d["embedding_api_key"] = resolve_api_key(emb_p, (d.get("embedding_api_key") or "").strip(), default_key)

    if include_rerank:
        rer_p = normalize_provider(d.get("rerank_provider"), "siliconflow")
        d["rerank_provider"] = rer_p
        rer_base = normalize_http_base(d.get("rerank_api_base") or "")
        if not rer_base:
            rer_base = legacy_base if rer_p == "siliconflow" and legacy_base else default_base_for(rer_p, ollama_base)
        d["rerank_api_base"] = rer_base
        d["rerank_api_key"] = resolve_api_key(rer_p, (d.get("rerank_api_key") or "").strip(), default_key)

    if include_llm:
        llm_p = normalize_provider(d.get("llm_provider"), "siliconflow")
        d["llm_provider"] = llm_p
        llm_base = normalize_http_base(d.get("llm_api_base") or "")
        if not llm_base:
            llm_base = legacy_base if llm_p == "siliconflow" and legacy_base else default_base_for(llm_p, ollama_base)
        d["llm_api_base"] = llm_base
        d["llm_api_key"] = resolve_api_key(llm_p, (d.get("llm_api_key") or "").strip(), default_key)

    vis_p = normalize_provider(d.get("vision_provider"), "siliconflow")
    d["vision_provider"] = vis_p
    vis_base = normalize_http_base(d.get("vision_api_base") or "")
    if not vis_base:
        vis_base = legacy_base if vis_p == "siliconflow" and legacy_base else default_base_for(vis_p, ollama_base)
    d["vision_api_base"] = vis_base
    d["vision_api_key"] = resolve_api_key(vis_p, (d.get("vision_api_key") or "").strip(), default_key)

    d["ollama_base_url"] = ollama_base
    return dict(d)

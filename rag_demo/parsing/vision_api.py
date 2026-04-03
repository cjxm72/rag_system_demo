"""
用 OpenAI 兼容多模态 Chat 解析图片（硅基 / OpenAI / Ollama，取决于请求中的 api_base）。
参数均由请求传入。
"""

from __future__ import annotations

import base64


def describe_image(
    image_base64: str,
    api_key: str,
    vision_model: str,
    api_base: str,
    prompt: str = "请提取图片中的全部文字与主要内容，便于后续检索或问答。",
) -> str:
    raw = image_base64.strip()
    if raw.startswith("data:"):
        raw = raw.split(",", 1)[-1].strip()
    try:
        b = base64.b64decode(raw, validate=True)
    except Exception:
        return "[图片 base64 无效]"
    data_url = f"data:image/jpeg;base64,{base64.b64encode(b).decode()}"
    if not api_key or not vision_model:
        return "[未配置 API Key 或视觉模型]"
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    if not api_base:
        return "[未配置 API Base URL]"
    llm = ChatOpenAI(base_url=api_base, api_key=api_key, model=vision_model, max_tokens=1024)
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]
    msg = HumanMessage(content=content)
    resp = llm.invoke([msg])
    return (resp.content or "").strip() or "[图片无有效内容]"


"""
解析 PDF 与图片为纯文本。PDF 用 PyMuPDF，图片用硅基流动视觉 API（需传入 api_key、vision_model、api_base）。
"""
import os
import base64
from typing import Optional

PDF_EXT = {".pdf"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def parse_pdf(path: str) -> str:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        parts = []
        for page in doc:
            parts.append(page.get_text())
        doc.close()
        return "\n\n".join(parts).strip() or "[PDF 无文本]"
    except ImportError:
        return "[未安装 PyMuPDF。请执行: pip install pymupdf]"
    except Exception as e:
        return f"[PDF 解析失败: {e}]"


def parse_image_with_vision(
    path: str,
    api_key: str,
    vision_model: str,
    api_base: str = "https://api.siliconflow.cn/v1",
    prompt: str = "请提取图片中的全部文字与主要内容，便于后续检索。",
) -> str:
    """用硅基流动视觉模型解析图片内容。"""
    if not api_key or not vision_model:
        return "[未配置 API Key 或视觉模型，无法解析图片]"
    try:
        with open(path, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode()
        data_url = f"data:image/jpeg;base64,{b64}"
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        llm = ChatOpenAI(base_url=api_base, api_key=api_key, model=vision_model, max_tokens=1024)
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        msg = HumanMessage(content=content)
        resp = llm.invoke([msg])
        return (resp.content or "").strip() or "[图片无有效内容]"
    except Exception as e:
        return f"[图片解析失败: {e}]"


def parse_file(
    path: str,
    api_key: Optional[str] = None,
    vision_model: Optional[str] = None,
    api_base: str = "https://api.siliconflow.cn/v1",
) -> str:
    path = os.path.abspath(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in PDF_EXT:
        return parse_pdf(path)
    if ext in IMAGE_EXT:
        return parse_image_with_vision(path, api_key or "", vision_model or "", api_base)
    # 纯文本
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        return f"[读取失败: {e}]"

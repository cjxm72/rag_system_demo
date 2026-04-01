"""
解析 PDF（OpenDataLoader PDF v2+）、Office 文档、图片与纯文本。
PDF 优先 OpenDataLoader，失败时回退 PyMuPDF（无 Java 环境时可用）。
"""

from __future__ import annotations

import glob
import os
import shutil
import tempfile
from typing import Optional

PDF_EXT = {".pdf"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
DOCX_EXT = {".docx"}


def parse_pdf_pymupdf(path: str) -> str:
    try:
        import fitz

        doc = fitz.open(path)
        parts = []
        for page in doc:
            parts.append(page.get_text())
        doc.close()
        return "\n\n".join(parts).strip() or "[PDF 无文本]"
    except ImportError:
        return "[未安装 PyMuPDF]"
    except Exception as e:
        return f"[PDF 解析失败: {e}]"


def parse_pdf_opendataloader(path: str) -> str:
    """OpenDataLoader PDF v2+：需系统安装 Java 11+。"""
    try:
        import opendataloader_pdf
    except ImportError:
        return ""

    path = os.path.abspath(path)
    tmp = tempfile.mkdtemp(prefix="odl_")
    try:
        opendataloader_pdf.convert(
            input_path=[path],
            output_dir=tmp,
            format="text",
            quiet=True,
        )
        files: list[str] = []
        for pattern in ("**/*.txt", "**/*.md", "**/*.markdown"):
            files.extend(glob.glob(os.path.join(tmp, pattern), recursive=True))
        files = sorted(set(files))
        if files:
            with open(files[0], "r", encoding="utf-8", errors="ignore") as f:
                return (f.read() or "").strip()
    except Exception:
        return ""
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return ""


def parse_pdf(path: str) -> str:
    text = parse_pdf_opendataloader(path)
    if text:
        return text
    return parse_pdf_pymupdf(path)


def parse_image_with_vision(
    path: str,
    api_key: str,
    vision_model: str,
    api_base: str,
    prompt: str = "请提取图片中的全部文字与主要内容，便于后续检索。",
) -> str:
    if not api_key or not vision_model:
        return "[未配置 API Key 或视觉模型，无法解析图片]"
    try:
        import base64

        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI

        with open(path, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode()
        data_url = f"data:image/jpeg;base64,{b64}"
        if not api_base:
            return "[未配置 API Base URL，无法解析图片]"
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


def parse_docx(path: str) -> str:
    try:
        import docx

        d = docx.Document(path)
        return "\n".join(p.text for p in d.paragraphs if p.text).strip() or "[DOCX 无文本]"
    except ImportError:
        return "[未安装 python-docx，无法解析 .docx]"
    except Exception as e:
        return f"[DOCX 解析失败: {e}]"


def parse_file(
    path: str,
    api_key: Optional[str] = None,
    vision_model: Optional[str] = None,
    api_base: str = "",
) -> str:
    path = os.path.abspath(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in PDF_EXT:
        return parse_pdf(path)
    if ext in IMAGE_EXT:
        return parse_image_with_vision(path, api_key or "", vision_model or "", api_base)
    if ext in DOCX_EXT:
        return parse_docx(path)
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        return f"[读取失败: {e}]"

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Literal, Optional, Tuple

from rag_demo.core.singleton import singleton


Domain = Literal["medical", "legal", "mixed", "general"]


@dataclass(frozen=True)
class TeamRouteDecision:
    domain: Domain
    reason: str


def _normalize_provider(settings: Dict[str, Any]) -> str:
    p = (settings.get("llm_provider") or "").lower().strip()
    if not p:
        raise ValueError("settings 缺少 llm_provider（需由前端 LocalStorage 传入）")
    if p not in ("siliconflow", "openai", "ollama"):
        raise ValueError(f"不支持的 llm_provider: {p}")
    return p


def _llm_http_credentials(settings: Dict[str, Any]) -> Tuple[str, str]:
    """返回 (base_url, api_key)，对应 OpenAI 兼容 Chat Completions。"""
    from rag_demo.core.provider_settings import finalize_settings

    finalize_settings(settings)
    base = (settings.get("llm_api_base") or "").rstrip("/")
    key = (settings.get("llm_api_key") or "").strip()
    if not base:
        raise ValueError("settings 缺少 llm_api_base（请先解析提供商或填写专用 Base）")
    if (settings.get("llm_provider") or "").lower() == "ollama":
        return base, key or "ollama"
    if not key:
        raise ValueError("settings 缺少 llm_api_key（硅基 / OpenAI 必填）")
    return base, key


@singleton
def build_openai_like_model(
    *,
    provider: str,
    model_id: str,
    api_key: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
) -> Any:
    # Agno 的 OpenAILike 可以对接所有 OpenAI-compatible 服务：
    # - 硅基流动：base_url=https://api.siliconflow.cn/v1, api_key=用户填写
    # - Ollama：base_url=http://localhost:11434/v1, api_key 任意（例如 "ollama"）
    try:
        from agno.models.openai.like import OpenAILike
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "未安装 agno。请在项目根目录执行 `uv sync` 或 `pip install agno` 后重试。原始错误: "
            + str(e)
        ) from e

    return OpenAILike(
        id=model_id,
        api_key=api_key or "not-provided",
        base_url=base_url,
        temperature=temperature,
        # 防止请求在网络波动时无限等待；Agno 会透传到 OpenAI-compatible 请求参数
        request_params={
            "max_tokens": max_tokens,
            "timeout": 120,
        },
    )


def _domain_prompts(domain: Domain) -> List[str]:
    # “提示词模拟微调”：用强约束的 domain instruction 模拟领域微调模型行为。
    common = [
        "你是一个严谨的助理。若检索内容不足以支撑结论，请明确说“不确定/缺少依据”，并给出需要补充的材料类型。",
        "优先引用检索上下文中的原句或近似复述，不要编造出处。",
        "回答结构尽量清晰：结论 -> 依据 -> 风险/例外 -> 下一步建议。",
    ]
    if domain == "medical":
        return common + [
            "你是【医疗领域】助理：面向临床/健康咨询场景，强调安全性与风险提示。",
            "避免给出具体处方/用药剂量等高风险建议；可给出一般性信息与就医建议。",
            "优先使用医学术语并解释给非专业读者；对不确定点要标注可能范围。",
        ]
    if domain == "legal":
        return common + [
            "你是【法律领域】助理：面向法律咨询场景，强调法条/要件/程序与适用前提。",
            "避免给出确定性结论替代律师意见；给出一般性分析与风险提示。",
            "优先给出：事实要点 -> 争点 -> 法律规则 -> 分析 -> 建议材料/行动。",
        ]
    if domain == "general":
        return common + ["你是通用助理：在医疗/法律之外的场景给出稳健回答。"]
    return common + ["你是协调者：需要在医疗与法律两种视角间做取舍或融合。"]


def _llm_route(question: str, settings: Dict[str, Any]) -> Tuple[TeamRouteDecision, List[str]]:
    """
    由“协调者模型”基于用户问题做路由，返回：
    - decision.domain: medical / legal / mixed / general
    - delegates: ["medical"] / ["legal"] / ["medical","legal"] / []
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    _normalize_provider(settings)
    base_url, api_key = _llm_http_credentials(settings)

    model = (settings.get("llm_model") or "").strip()
    if not model:
        raise ValueError("settings 缺少 llm_model（协调者路由与回答共用该模型 id）")

    llm = ChatOpenAI(base_url=base_url, api_key=api_key, model=model, temperature=0.0, max_tokens=200)
    sys = SystemMessage(
        content=(
            "你是一个路由器。根据用户问题判断属于哪个领域，并决定需要委派哪些专家。\n"
            "输出必须是严格 JSON（不要 markdown/解释），格式：\n"
            "{\"domain\":\"medical|legal|mixed|general\",\"delegates\":[\"medical\",\"legal\"],\"reason\":\"一句话原因\"}\n"
            "规则：\n"
            "- medical: 医疗健康/诊疗相关 -> delegates 包含 medical\n"
            "- legal: 法律合规/诉讼/合同等 -> delegates 包含 legal\n"
            "- mixed: 同时涉及医疗与法律 -> delegates 同时包含 medical 与 legal\n"
            "- general: 其他 -> delegates 为空数组\n"
        )
    )
    msg = HumanMessage(content=(question or "").strip())

    raw = llm.invoke([sys, msg]).content or ""
    import json as _json

    try:
        obj = _json.loads(raw)
    except Exception:
        return TeamRouteDecision(domain="general", reason="协调者路由输出非 JSON，已回退 general"), []

    domain = str(obj.get("domain") or "").strip()
    reason = str(obj.get("reason") or "").strip() or "协调者模型路由"
    delegates = obj.get("delegates") or []
    if not isinstance(delegates, list):
        delegates = []
    delegates = [str(x).strip() for x in delegates if str(x).strip() in {"medical", "legal"}]
    delegates = list(dict.fromkeys(delegates))
    if domain not in {"medical", "legal", "mixed", "general"}:
        domain = "general"
    return TeamRouteDecision(domain=domain, reason=reason), delegates


@singleton
def build_domain_team(
    *,
    provider: str,
    base_url: str,
    api_key: str,
    model_id_medical: str,
    model_id_legal: str,
    model_id_coordinator: str,
    temperature: float,
    max_tokens: int,
) -> Any:
    try:
        from agno.agent import Agent
        from agno.team import Team
        from agno.team.mode import TeamMode
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "未安装 agno。请在项目根目录执行 `uv sync` 或 `pip install agno` 后重试。原始错误: "
            + str(e)
        ) from e

    medical_agent = Agent(
        id="medical-agent",
        name="Medical Agent",
        role="医疗领域回答与风险提示",
        model=build_openai_like_model(
            provider=provider,
            model_id=model_id_medical,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        instructions=_domain_prompts("medical"),
        markdown=True,
    )
    legal_agent = Agent(
        id="legal-agent",
        name="Legal Agent",
        role="法律领域分析与合规建议",
        model=build_openai_like_model(
            provider=provider,
            model_id=model_id_legal,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        instructions=_domain_prompts("legal"),
        markdown=True,
    )

    coordinator = Team(
        id="coordinator-team",
        name="Coordinator Team",
        role="路由与融合医疗/法律输出",
        members=[medical_agent, legal_agent],
        model=build_openai_like_model(
            provider=provider,
            model_id=model_id_coordinator,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        mode=TeamMode.coordinate,
        instructions=[
            *_domain_prompts("mixed"),
            "你负责决定是否委派给 Medical Agent、Legal Agent 或两者，并在必要时融合其结论。",
            "如果问题明显属于单一领域，只委派一个成员并直接给出最终答复。",
            "如果问题涉及医疗与法律的交叉（例如医疗纠纷、医疗合同、病历证据），委派两个成员并融合输出。",
            "输出需包含一个“路由说明”小节：说明你选择了哪个成员/是否混合，以及原因（简短即可）。",
        ],
        markdown=True,
    )
    # 返回 (medical_agent, legal_agent, coordinator_team)
    return medical_agent, legal_agent, coordinator


def run_team_answer(
    *,
    question: str,
    context: str,
    settings: Dict[str, Any],
    stream: bool = False,
    route_question: Optional[str] = None,
) -> Tuple[str, TeamRouteDecision, Optional[Iterator[str]]]:
    """
    返回 (answer, decision, stream_iter)
    - stream=False: stream_iter 为 None
    - stream=True: answer 为空字符串，stream_iter 逐步产出文本
    """
    provider = _normalize_provider(settings)
    base_url, api_key = _llm_http_credentials(settings)

    temperature = float(settings.get("temperature", 0.7))
    max_tokens = int(settings.get("max_tokens", 2000))

    # 允许在 settings 里覆盖三套“模拟微调模型”的 model id；不填则回退到 llm_model
    fallback_model = (settings.get("llm_model") or "").strip()
    if not fallback_model:
        raise ValueError("settings 缺少 llm_model（需由前端 LocalStorage 传入）")
    model_id_medical = settings.get("llm_model_medical") or fallback_model
    model_id_legal = settings.get("llm_model_legal") or fallback_model
    model_id_coordinator = fallback_model

    # 由协调者模型做“路由+委派”决策
    decision, delegates = _llm_route(route_question or question, settings)

    medical_agent, legal_agent, team = build_domain_team(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model_id_medical=model_id_medical,
        model_id_legal=model_id_legal,
        model_id_coordinator=model_id_coordinator,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # 将 RAG 上下文拼入输入（Agno 会把 input 作为 user message）
    team_input = (
        "【用户问题】\n"
        + (question or "").strip()
        + "\n\n【检索上下文】\n"
        + (context or "").strip()
        + "\n\n【引用规则】\n"
        + "检索片段已标注 [来源i] 与 doc_id；回答时若依据某条片段，请在相关句子后标注对应编号（如[来源1]）。"
        + "不要在回答末尾自行列出「引用文档」清单，系统会根据 doc_id 从数据库反查后自动追加。"
    )

    if not stream:
        timeout_s = int(settings.get("request_timeout_s") or 180)
        retries = int(settings.get("request_retries") or 2)

        def _run_once(timeout_one: int):
            ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            fut = ex.submit(lambda: team.run(team_input, stream=False))
            try:
                return fut.result(timeout=timeout_one)
            except concurrent.futures.TimeoutError as e:
                fut.cancel()
                ex.shutdown(wait=False, cancel_futures=True)
                raise RuntimeError(f"LLM 调用超时（>{timeout_one}s）") from e
            finally:
                # 正常情况：等待线程结束并回收资源
                ex.shutdown(wait=True, cancel_futures=False)

        last_err: Exception | None = None
        # 将总超时拆成多次尝试，便于对 429 做退避重试，避免一次卡死拖满 timeout_s
        per_try = max(30, min(90, timeout_s))
        for attempt in range(retries + 1):
            try:
                # 按 delegates 决定实际调用哪个专家/是否走 Team 协调
                if delegates == ["medical"]:
                    out = medical_agent.run(team_input, stream=False)
                elif delegates == ["legal"]:
                    out = legal_agent.run(team_input, stream=False)
                else:
                    out = _run_once(per_try)
                return getattr(out, "content", "") or "", decision, None
            except Exception as e:
                last_err = e
                msg = str(e)
                if "429" in msg or "rate limit" in msg.lower() or "TPM" in msg:
                    time.sleep(min(2**attempt, 8))
                    continue
                # 超时也允许重试一次（网络抖动/服务不稳定）
                if "超时" in msg and attempt < retries:
                    time.sleep(1)
                    continue
                break
        raise RuntimeError(f"{last_err}。可在 settings.request_timeout_s / request_retries 调整。")

    def _iter_text() -> Iterator[str]:
        stream_iter = team.run(team_input, stream=True)
        for chunk in stream_iter:
            text = getattr(chunk, "content", "") or ""
            if text:
                yield text

    return "", decision, _iter_text()


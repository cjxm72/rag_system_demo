"""
对话流程：检索（支持多知识组 + 主/次组权重）→ 生成。
LLM 仅支持硅基流动 或 Ollama（OpenAI 兼容 URL）。
"""
from typing import TypedDict, Annotated, Sequence, Dict, Any, List

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.config_loader import load_config
from src.rag_system import query as rag_query

config = load_config()


class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]
    question: str
    groups: List[Dict[str, Any]]  # [{id, priority}]
    settings: dict  # api_key, embedding_model, reranker_model, api_base, llm_provider, llm_model, ollama_base_url


def _get_settings(state: AgentState) -> dict:
    return state.get("settings") or {}


def initialize_llm(settings: dict):
    """仅支持 siliconflow 或 ollama；ollama 用 OpenAI 兼容 base_url。"""
    from langchain_openai import ChatOpenAI

    provider = (settings.get("llm_provider") or "siliconflow").lower()
    api_base = settings.get("api_base") or "https://api.siliconflow.cn/v1"
    model = settings.get("llm_model") or "Pro/deepseek-ai/DeepSeek-V3.2"
    temperature = float(settings.get("temperature", 0.7))
    max_tokens = int(settings.get("max_tokens", 2000))

    if provider == "ollama":
        base = (settings.get("ollama_base_url") or "http://localhost:11434/v1").rstrip("/")
        return ChatOpenAI(
            base_url=base,
            api_key="ollama",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    # 硅基流动
    api_key = settings.get("api_key") or ""
    if not api_key:
        raise ValueError("请在前端设置中填写硅基流动 API 密钥。")
    return ChatOpenAI(
        base_url=api_base,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def retrieve_node(state: AgentState) -> dict:
    question = state.get("question") or ""
    selected_groups = state.get("groups") or []
    s = _get_settings(state)
    api_key = s.get("api_key") or ""
    embedding_model = s.get("embedding_model") or "BAAI/bge-large-zh-v1.5"
    reranker_model = s.get("reranker_model") or "BAAI/bge-reranker-v2-m3"
    api_base = s.get("api_base") or "https://api.siliconflow.cn/v1"
    top_k = config.vector_store.similarity_top_k
    rerank_n = config.vector_store.rerank_top_n

    context = rag_query(
        question=question,
        selected_groups=selected_groups,
        api_key=api_key,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        api_base=api_base,
        rerank_top_n=rerank_n,
        similarity_top_k=top_k * 2,
    )
    system_prompt = (
        config.prompts.system
        + "\n\n--- 检索到的相关文档 ---\n"
        + context
        + "\n\n--- 请严格根据以上文档内容回答问题，不要编造 ---"
    )
    return {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]
    }


def generate_node(state: AgentState) -> dict:
    s = _get_settings(state)
    llm = initialize_llm(s)
    messages = state.get("messages") or []
    response = llm.invoke(messages)
    return {"messages": [response]}


def create_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile(checkpointer=MemorySaver())

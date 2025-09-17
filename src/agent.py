from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage

from config_loader import load_config

config = load_config()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]
    question: str


def initialize_llm():
    """根据配置初始化LLM"""
    llm_config = config.models["llm"]  # 从配置获取LLM设置

    if llm_config.provider == "local":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            base_url="http://localhost:8001/v1",
            api_key="EMPTY",
            model="local",
            temperature=0.1,
            max_tokens=2000
        )
    elif llm_config.provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=llm_config.name,
            api_key=llm_config.api_key
        )
    else:
        raise ValueError(f"不支持的模型提供商: {llm_config.provider}")


def retrieve_node(state: AgentState,rag_tool) -> dict:
    """
    调用 RAGTool 获取相关文档片段
    返回构造好的 messages（含系统提示 + 上下文 + 用户问题）
    """
    question = state["question"]
    print(f"\n📥 [检索节点] 用户问题: {question}")

    # 1️⃣ 调用 RAGTool 获取上下文（此时会打印检索结果！）
    context = rag_tool._run(question)

    # 2️⃣ 构造系统提示：告诉 Qwen 必须根据上下文回答
    system_prompt = (
            config.prompts.system +  # 原有系统提示（如“你是一个企业秘书...”）
            "\n\n--- 检索到的相关文档 ---\n" +
            context +
            "\n\n--- 请严格根据以上文档内容回答问题，不要编造 ---"
    )
    system_message = SystemMessage(content=system_prompt)

    # 3️⃣ 用户问题
    human_message = HumanMessage(content=question)

    # 4️⃣ 返回新的 messages 列表（覆盖旧的！）
    return {
        "messages": [system_message, human_message]
    }


_PROMPT_LOGGED = False


def generate_node(state: AgentState) -> dict:
    global _PROMPT_LOGGED
    llm = initialize_llm()
    messages = state["messages"]

    # 提取用户问题
    user_question = ""
    context = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            user_question = msg.content
        elif isinstance(msg, SystemMessage):
            # 从系统消息里提取上下文（可选）
            if "--- 检索到的相关文档 ---" in msg.content:
                parts = msg.content.split("--- 检索到的相关文档 ---")
                if len(parts) > 1:
                    context = parts[1].split("--- 请严格根据")[0].strip()

    response = llm.invoke(messages)

    # 第一次打印系统提示词
    if not _PROMPT_LOGGED:
        print("\n" + "=" * 60)
        print("🤖 系统提示词（仅首次显示）")
        for msg in messages:
            if isinstance(msg, SystemMessage):
                print(msg.content)
                break
        print("=" * 60 + "\n")
        _PROMPT_LOGGED = True

    # 每次只打印简洁问答
    print(f"📥 [用户提问] {user_question}")
    if context:
        print(f"📄 [参考上下文] {context}")
    print(f"✅ [AI回复] {response.content}\n")

    return {"messages": [response]}


def create_workflow(rag_tool):
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", lambda state: retrieve_node(state, rag_tool))  # 第一步：检索
    workflow.add_node("generate", generate_node)  # 第二步：生成

    workflow.set_entry_point("retrieve")  # 入口是检索
    workflow.add_edge("retrieve", "generate")  # 检索完 → 生成
    workflow.add_edge("generate", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

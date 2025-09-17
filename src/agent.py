from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent

from langgraph.graph import StateGraph,END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage

from config_loader import load_config

config = load_config()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],lambda x,y:x+y]
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

def create_agent(rag_tool):
    llm = initialize_llm()

    tools = [rag_tool]
    llm_with_tools = llm.bind_tools(tools)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=config.prompts.system),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{question}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm,tools,prompt)
    return AgentExecutor(agent=agent, tools=tools,verbose=True)

def create_workflow(agent_executor):


    workflow = StateGraph(AgentState)
    workflow.add_node("agent", lambda state: agent_executor.invoke(state))
    workflow.add_node("human",lambda state: {"messages": [HumanMessage(content=state["question"])]})

    workflow.set_entry_point("human")

    workflow.add_edge("human", "agent")
    workflow.add_edge("agent", END)

    memory = MemorySaver()

    return workflow.compile(checkpointer=memory)






















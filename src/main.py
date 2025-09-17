import os
# 强制离线
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import sys
from fastapi import FastAPI, HTTPException
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from nltk.corpus.reader import documents
from pydantic import BaseModel
from typing import List
from rag_system import RAGSystem, RAGTool
from agent import create_agent, create_workflow
from config_loader import load_config



current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 切换工作目录到项目根目录
os.chdir(project_root)
# 将项目根目录加入 sys.path，确保模块导入正常
if project_root not in sys.path:
    sys.path.insert(0, project_root)

app = FastAPI(title="企业数据管理秘书系统")


config = load_config()

rag_system = RAGSystem()

@app.on_event("startup")
async def startup_event():
    from llama_index.core import SimpleDirectoryReader

    if not os.path.exists("./data/documents"):
        os.makedirs("./data/documents")
        print("请将内容放入./data/documents中")
        return
    document = SimpleDirectoryReader("./data/documents").load_data()
    rag_system.load_document(document)

class QueryRequest(BaseModel):
    question: str
    thread_id: str = "default"

class QueryResponse(BaseModel):
    answer: str
    thread_id: str

rag_tool = RAGTool(rag_system=rag_system)
agent = create_agent(rag_tool)
workflow = create_workflow(agent)
@app.post("/query",response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        result = workflow.invoke(
            {"question": request.question},
            {"configurable": {"thread_id": request.thread_id}}
        )
        return   QueryResponse(
            answer=result["messages"][-1].content,
            thread_id=request.thread_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

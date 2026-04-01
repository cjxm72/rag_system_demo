"""python -m rag_demo"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("rag_demo.api.main:app", host="0.0.0.0", port=8001)

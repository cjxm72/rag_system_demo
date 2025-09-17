下载uv库后在在终端使用uv sync命令加载依赖（请先创建虚拟环境.venv，建议使用UV,不清楚conda是否能兼容UV）

data/documents下放入你想要传入知识库的文档，目前不支持pdf，jpg等图像识别，只支持.md .text .json .excel 等文档类识别，后续考虑添加图像识别

在models下终端运行git clone https://www.modelscope.cn/Qwen/Qwen2.5-1.5B-Instruct-GGUF.git 来下载模型,但我们只需要qwen2.5-1.5b-instruct-q4_k_m.gguf模型，所以你可以只手动下载这一个模型与配置文件


在终端运行 LD_LIBRARY_PATH=./server/bin ./server/bin/llama-server -m ./models/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf --port 8001 --ctx-size 4096

最后直接运行src/mian.py即可
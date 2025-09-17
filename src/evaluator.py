from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset
from typing import List, Dict


class RAGEvaluator:
    """RAG系统评估模块"""

    def __init__(self):
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

    def evaluate(self, questions: List[str], answers: List[str], contexts: List[List[str]]):
        """
        执行RAG评估

        参数:
            questions: 问题列表
            answers: 对应答案列表
            contexts: 每个答案对应的上下文列表

        返回:
            各指标得分字典
        """
        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts
        })

        result = evaluate(dataset, self.metrics)
        return result
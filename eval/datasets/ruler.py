# -*- coding: utf-8 -*-
"""
RULER 数据集评估

RULER (Rule-based Understanding of Long-context Evaluation Resource)
是一个测试长上下文理解能力的数据集
"""

import json
from typing import Any, Dict, List
from eval.datasets.base import BaseBenchmark, BenchmarkConfig
from eval.utils import chunk_text_smartly
from eval.utils.metrics import evaluate_answer_ruler


class RULERBenchmark(BaseBenchmark):
    """RULER 评估基准"""
    
    # 数据集特定的 system prompts
    SYSTEM_PROMPTS = {
        "vt": "Memorize and track the chain(s) of variable assignment hidden in the following text.",
        "qa_1": "",
        "qa_2": "",
        "niah_single_1": "A special magic number is hidden within the following text. Make sure to memorize it. I will quiz you about the number afterwards.",
        "niah_single_2": "A special magic number is hidden within the following text. Make sure to memorize it. I will quiz you about the number afterwards.",
        "niah_single_3": "A special magic uuid is hidden within the following text. Make sure to memorize it. I will quiz you about the uuid afterwards.",
        "niah_multivalue": "",
        "niah_multiquery": "Some special magic numbers are hidden within the following text. You only need to memorize the special magic numbers. I will quiz you about the numbers afterwards.",
        "niah_multikey_1": "",
        "niah_multikey_2": "",
        "niah_multikey_3": "",
        "cwe": "Below is a numbered list of words. You only need to memorize the numbers that all words appear rather then make a abstract. I will quiz you about the numbers afterwards. Ignore the prompt below that asks you to summarize.",
        "fwe": "Read the following coded text and track the frequency of each coded word. Memorize the numbers that the words appear, I will quiz you about the numbers afterwards.",
    }
    
    def __init__(self, config: BenchmarkConfig, dataset_name: str = ""):
        super().__init__(config)
        self.dataset_name = dataset_name
    
    def load_data(self) -> List[Dict[str, Any]]:
        """加载 RULER JSONL 数据"""
        data_all = []
        
        with open(self.config.data_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    data_all.append({
                        "index": idx,
                        "context": item.get("context", ""),
                        "example": item.get("example", ""),
                        "instruction": item.get("instruction", ""),
                        "question": item.get("question", ""),
                        "outputs": item.get("outputs", []),
                    })
                except json.JSONDecodeError as e:
                    print(f"跳过无效的 JSON 行 {idx}: {e}")
                    continue
        
        return data_all
    
    def prepare_chunks(self, sample: Dict[str, Any]) -> List[str]:
        """
        准备 context 分块
        对于 RULER，context 通常很长，需要智能切分
        """
        context = sample.get("context", "")
        if not context:
            return []
        
        # 添加 system prompt（如果有）
        system_prompt = self._get_system_prompt()
        if system_prompt:
            context = f"{system_prompt}\n\n{context}"
        
        return chunk_text_smartly(
            context,
            max_tokens=self.config.chunk_size
        )
    
    def extract_question(self, sample: Dict[str, Any]) -> str:
        """
        提取问题
        RULER 的问题可能包含 example 和 question
        """
        example = sample.get("example", "")
        question = sample.get("question", "")
        
        if example and question:
            return f"{example}\n\n{question}"
        elif question:
            return question
        else:
            return ""
    
    def extract_ground_truth(self, sample: Dict[str, Any]) -> List[str]:
        """提取标准答案"""
        outputs = sample.get("outputs", [])
        if isinstance(outputs, list):
            return [str(o) for o in outputs if o]
        return [str(outputs)] if outputs else []
    
    def compute_metrics(
        self, 
        predictions: List[str], 
        ground_truths: List[List[str]]
    ) -> Dict[str, float]:
        """
        计算准确率（Accuracy）
        RULER 主要使用精确匹配
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("预测数量与标准答案数量不匹配")
        
        correct = 0
        for pred, gts in zip(predictions, ground_truths):
            if evaluate_answer_ruler(pred, gts):
                correct += 1
        
        accuracy = correct / len(predictions) if predictions else 0.0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(predictions),
        }
    
    def _get_system_prompt(self) -> str:
        """获取数据集特定的 system prompt"""
        if not self.dataset_name:
            return ""
        
        # 尝试精确匹配
        if self.dataset_name in self.SYSTEM_PROMPTS:
            return self.SYSTEM_PROMPTS[self.dataset_name]
        
        # 尝试模糊匹配（去掉数字后缀）
        base_name = self.dataset_name.split('_')[0]
        for key, prompt in self.SYSTEM_PROMPTS.items():
            if key.startswith(base_name):
                return prompt
        
        return ""


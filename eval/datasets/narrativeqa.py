# -*- coding: utf-8 -*-
"""
NarrativeQA 数据集评估

NarrativeQA 是一个叙事阅读理解数据集，包含长文档和相关问题
"""

from typing import Any, Dict, List
from eval.datasets.base import BaseBenchmark, BenchmarkConfig
from eval.utils import chunk_text_by_sentences, compute_metrics


class NarrativeQABenchmark(BaseBenchmark):
    """NarrativeQA 评估基准"""
    
    def load_data(self) -> List[Dict[str, Any]]:
        """加载 NarrativeQA 数据集"""
        from datasets import load_dataset
        
        # 加载数据集（支持 parquet 格式或 HuggingFace 数据集名称）
        if self.config.data_path.endswith('.parquet') or '/' in self.config.data_path:
            # 如果是路径，使用 parquet 格式加载
            dataset = load_dataset("parquet", data_dir=self.config.data_path, split="test")
        else:
            # 如果是数据集名称，直接加载
            dataset = load_dataset(self.config.data_path, split="test")
        
        data_all = []
        for idx, item in enumerate(dataset):
            document = item.get("document", {})
            question = item.get("question", {})
            answers = item.get("answers", [])
            
            data_all.append({
                "index": idx,
                "document_id": document.get("id", f"doc-{idx}"),
                "document_text": document.get("text", ""),
                "question_text": question.get("text", ""),
                "answers": [ans.get("text", "") for ans in answers if isinstance(ans, dict)],
            })
        
        return data_all
    
    def prepare_chunks(self, sample: Dict[str, Any]) -> List[str]:
        """将文档文本分块（按句子切分以保持连贯性）"""
        document_text = sample.get("document_text", "")
        if not document_text:
            return []
        
        return chunk_text_by_sentences(
            document_text,
            max_tokens=self.config.chunk_size
        )
    
    def extract_question(self, sample: Dict[str, Any]) -> str:
        """提取问题"""
        return sample.get("question_text", "")
    
    def extract_ground_truth(self, sample: Dict[str, Any]) -> List[str]:
        """提取标准答案"""
        answers = sample.get("answers", [])
        return [str(a) for a in answers if a]
    
    def compute_metrics(
        self, 
        predictions: List[str], 
        ground_truths: List[List[str]]
    ) -> Dict[str, float]:
        """计算 F1 指标"""
        return compute_metrics(
            predictions, 
            ground_truths, 
            metrics=["f1"]
        )


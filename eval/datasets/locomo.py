# -*- coding: utf-8 -*-
"""
LoCoMo 数据集评估

LoCoMo 是一个长对话记忆数据集，测试多轮对话中的记忆能力
"""

import re
import json
from typing import Any, Dict, List, Optional, Tuple
from eval.datasets.base import BaseBenchmark, BenchmarkConfig
from eval.utils.metrics import compute_locomo_metrics


class LoCoMoBenchmark(BaseBenchmark):
    """LoCoMo 评估基准"""
    
    def load_data(self) -> List[Dict[str, Any]]:
        """加载 LoCoMo JSON 数据"""
        with open(self.config.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and "samples" in data:
            return data["samples"]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError("Unrecognized LoCoMo JSON format")
    
    def prepare_chunks(self, sample: Dict[str, Any]) -> List[str]:
        """
        将对话分成多个 session chunks
        每个 session 作为一个独立的记忆单元
        """
        conv = sample.get("conversation", {})
        sessions = self._extract_sessions(conv)
        
        chunks = []
        for idx, timestamp, turns, session_summary in sessions:
            chunk = self._session_to_text(idx, timestamp, turns, session_summary)
            chunks.append(chunk)
        
        return chunks
    
    def extract_question(self, sample: Dict[str, Any]) -> str:
        """
        LoCoMo 有多个问题，这里返回第一个问题
        实际使用中可能需要遍历所有问题
        """
        qa_items = sample.get("qa", [])
        if qa_items:
            return qa_items[0].get("question", "")
        return ""
    
    def extract_ground_truth(self, sample: Dict[str, Any]) -> List[str]:
        """提取标准答案"""
        qa_items = sample.get("qa", [])
        if qa_items:
            answer = qa_items[0].get("answer", "")
            return [answer] if answer else []
        return []
    
    def compute_metrics(
        self, 
        predictions: List[str], 
        ground_truths: List[List[str]]
    ) -> Dict[str, float]:
        """计算 F1 和 BLEU-1 指标（LoCoMo 特定）"""
        return compute_locomo_metrics(predictions, ground_truths)
    
    def run(self) -> Dict[str, float]:
        """
        重写 run 方法以处理多个 QA 对
        LoCoMo 每个样本可能有多个问题
        """
        print(f"正在加载数据集: {self.config.data_path}")
        self.data = self.load_data()
        
        if self.config.max_samples:
            self.data = self.data[:self.config.max_samples]
        
        print(f"加载了 {len(self.data)} 个样本")
        
        # 初始化 Agent
        print("正在初始化 GAM Agent...")
        memory_agent, research_agent = self._setup_agents()
        
        # 运行评估
        print("开始评估...")
        self.predictions = []
        ground_truths = []
        
        for idx, sample in enumerate(self.data):
            if self.config.verbose:
                print(f"\n处理样本 {idx + 1}/{len(self.data)}")
            
            try:
                # 准备chunks并记忆
                chunks = self.prepare_chunks(sample)
                for chunk in chunks:
                    memory_agent.memorize(chunk)
                
                # 处理所有 QA 对
                qa_items = sample.get("qa", [])
                for qa in qa_items:
                    question = qa.get("question", "")
                    answer = qa.get("answer", "")
                    
                    if not question:
                        continue
                    
                    # 研究
                    research_output = research_agent.research(question)
                    
                    prediction = research_output.integrated_memory
                    self.predictions.append(prediction)
                    ground_truths.append([answer] if answer else [""])
                    
                    if self.config.verbose:
                        print(f"问题: {question[:80]}...")
                        print(f"预测: {prediction[:80]}...")
                        print(f"标准答案: {answer[:80]}...")
            
            except Exception as e:
                print(f"处理样本 {idx} 时出错: {e}")
                continue
        
        # 计算指标
        print("\n计算评估指标...")
        self.results = self.compute_metrics(self.predictions, ground_truths)
        
        # 保存结果
        if self.config.save_predictions:
            self._save_results()
        
        return self.results
    
    def _extract_sessions(
        self, 
        conv_obj: Dict[str, Any]
    ) -> List[Tuple[int, str, List[Dict[str, Any]], Optional[str]]]:
        """提取 session 信息"""
        sessions = []
        for k, v in conv_obj.items():
            m = re.match(r'^session_(\d+)$', k)
            if not (m and isinstance(v, list)):
                continue
            
            idx = int(m.group(1))
            timestamp = conv_obj.get(f"session_{idx}_date_time", "")
            summary = conv_obj.get(f"session_{idx}_summary", None)
            
            if summary and not isinstance(summary, str):
                summary = None
            
            sessions.append((idx, timestamp, v, summary))
        
        sessions.sort(key=lambda x: x[0])
        return sessions
    
    def _session_to_text(
        self,
        idx: int,
        timestamp: str,
        turns: List[Dict[str, Any]],
        session_summary: Optional[str]
    ) -> str:
        """将 session 转换为文本"""
        lines = [
            f"=== SESSION {idx} - Dialogue Time(available to answer questions): {timestamp} ===",
            ""
        ]
        
        for turn in turns:
            speaker = turn.get("speaker", "Unknown")
            dia_id = turn.get("dia_id", "")
            text = turn.get("text", "")
            lines.append(f"{speaker} ({dia_id}): {text}")
        
        if session_summary:
            lines.append("")
            lines.append(f"Session {idx} summary: {session_summary}")
        
        return "\n".join(lines)


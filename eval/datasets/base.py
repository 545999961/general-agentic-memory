# -*- coding: utf-8 -*-
"""
评估基准基类

所有数据集评估都应继承此基类
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    """评估配置"""
    # 数据路径
    data_path: str
    
    # 模型配置
    generator_type: str = "openai"  # "openai" or "vllm"
    model_name: str = "gpt-4"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    # 检索器配置
    retriever_type: str = "dense"  # "index", "bm25", "dense"
    embedding_model: Optional[str] = None
    
    # 评估配置
    max_samples: Optional[int] = None
    num_workers: int = 4
    chunk_size: int = 2000
    top_k: int = 5
    
    # 输出配置
    output_dir: str = "outputs"
    save_predictions: bool = True
    verbose: bool = True


class BaseBenchmark(ABC):
    """评估基准基类"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.data = []
        self.predictions = []
        self.results = {}
    
    @abstractmethod
    def load_data(self) -> List[Dict[str, Any]]:
        """加载数据集"""
        pass
    
    @abstractmethod
    def prepare_chunks(self, sample: Dict[str, Any]) -> List[str]:
        """准备待记忆的文本块"""
        pass
    
    @abstractmethod
    def extract_question(self, sample: Dict[str, Any]) -> str:
        """提取问题"""
        pass
    
    @abstractmethod
    def extract_ground_truth(self, sample: Dict[str, Any]) -> List[str]:
        """提取标准答案"""
        pass
    
    @abstractmethod
    def compute_metrics(self, predictions: List[str], ground_truths: List[List[str]]) -> Dict[str, float]:
        """计算评估指标"""
        pass
    
    def run(self) -> Dict[str, float]:
        """
        运行完整评估流程
        
        Returns:
            评估结果字典
        """
        # 1. 加载数据
        print(f"正在加载数据集: {self.config.data_path}")
        self.data = self.load_data()
        
        if self.config.max_samples:
            self.data = self.data[:self.config.max_samples]
        
        print(f"加载了 {len(self.data)} 个样本")
        
        # 2. 初始化 Agent
        print("正在初始化 GAM Agent...")
        memory_agent, research_agent = self._setup_agents()
        
        # 3. 运行评估
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
                
                # 提取问题并研究
                question = self.extract_question(sample)
                research_output = research_agent.research(question)
                
                prediction = research_output.integrated_memory
                self.predictions.append(prediction)
                
                # 提取标准答案
                gt = self.extract_ground_truth(sample)
                ground_truths.append(gt)
                
                if self.config.verbose:
                    print(f"预测: {prediction[:100]}...")
                    print(f"标准答案: {gt[0][:100] if gt else 'N/A'}...")
            
            except Exception as e:
                print(f"处理样本 {idx} 时出错: {e}")
                self.predictions.append("")
                ground_truths.append([""])
        
        # 4. 计算指标
        print("\n计算评估指标...")
        self.results = self.compute_metrics(self.predictions, ground_truths)
        
        # 5. 保存结果
        if self.config.save_predictions:
            self._save_results()
        
        return self.results
    
    def _setup_agents(self):
        """设置 Memory 和 Research Agent"""
        from gam import (
            MemoryAgent,
            ResearchAgent,
            OpenAIGenerator,
            VLLMGenerator,
            OpenAIGeneratorConfig,
            VLLMGeneratorConfig,
            InMemoryMemoryStore,
            InMemoryPageStore,
            IndexRetriever,
            BM25Retriever,
            DenseRetriever,
            IndexRetrieverConfig,
            BM25RetrieverConfig,
            DenseRetrieverConfig,
        )
        
        # 创建 Generator
        if self.config.generator_type == "openai":
            gen_config = OpenAIGeneratorConfig(
                model_name=self.config.model_name,
                api_key=self.config.api_key,
                base_url=self.config.api_base,
            )
            generator = OpenAIGenerator(gen_config.__dict__)
        elif self.config.generator_type == "vllm":
            gen_config = VLLMGeneratorConfig(
                model_name=self.config.model_name,
            )
            generator = VLLMGenerator(gen_config.__dict__)
        else:
            raise ValueError(f"Unknown generator type: {self.config.generator_type}")
        
        # 创建存储
        memory_store = InMemoryMemoryStore()
        page_store = InMemoryPageStore()
        
        # 创建检索器
        retrievers = {}
        if self.config.retriever_type == "index":
            retriever_config = IndexRetrieverConfig()
            retriever = IndexRetriever(retriever_config.__dict__)
            retriever.build(page_store)
            retrievers["page_index"] = retriever
        elif self.config.retriever_type == "bm25":
            retriever_config = BM25RetrieverConfig()
            retriever = BM25Retriever(retriever_config.__dict__)
            retriever.build(page_store)
            retrievers["keyword"] = retriever
        elif self.config.retriever_type == "dense":
            retriever_config = DenseRetrieverConfig(
                model_name=self.config.embedding_model or "BAAI/bge-base-en-v1.5"
            )
            retriever = DenseRetriever(retriever_config.__dict__)
            retriever.build(page_store)
            retrievers["vector"] = retriever
        else:
            raise ValueError(f"Unknown retriever type: {self.config.retriever_type}")
        
        # 创建 Agent
        memory_agent = MemoryAgent(
            generator=generator,
            memory_store=memory_store,
            page_store=page_store,
        )
        
        research_agent = ResearchAgent(
            generator=generator,
            page_store=page_store,
            memory_store=memory_store,
            retrievers=retrievers,
        )
        
        return memory_agent, research_agent
    
    def _save_results(self):
        """保存评估结果"""
        import os
        import json
        from datetime import datetime
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(
            self.config.output_dir,
            f"{self.__class__.__name__}_{timestamp}.json"
        )
        
        output = {
            "config": {
                "data_path": self.config.data_path,
                "generator_type": self.config.generator_type,
                "model_name": self.config.model_name,
                "retriever_type": self.config.retriever_type,
                "num_samples": len(self.data),
            },
            "metrics": self.results,
            "predictions": [
                {
                    "prediction": pred,
                    "ground_truth": self.extract_ground_truth(sample),
                }
                for pred, sample in zip(self.predictions, self.data)
            ]
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存到: {result_file}")


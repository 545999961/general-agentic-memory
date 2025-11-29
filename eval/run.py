#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAM Evaluation Suite - 统一评估入口

使用示例：
    # HotpotQA
    python -m eval.run --dataset hotpotqa --data-path data/hotpotqa.json
    
    # NarrativeQA
    python -m eval.run --dataset narrativeqa --data-path narrativeqa --max-samples 100
    
    # LoCoMo
    python -m eval.run --dataset locomo --data-path data/locomo.json
    
    # RULER
    python -m eval.run --dataset ruler --data-path data/ruler.jsonl --dataset-name niah_single_1
"""

import argparse
import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from eval.datasets import (
    HotpotQABenchmark,
    NarrativeQABenchmark,
    LoCoMoBenchmark,
    RULERBenchmark,
)
from eval.datasets.base import BenchmarkConfig


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="GAM Framework 评估工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 评估 HotpotQA（使用 OpenAI GPT-4）
  python -m eval.run --dataset hotpotqa --data-path data/hotpotqa.json \\
      --generator openai --model gpt-4 --api-key YOUR_API_KEY
  
  # 评估 NarrativeQA（使用本地 VLLM 模型）
  python -m eval.run --dataset narrativeqa --data-path narrativeqa \\
      --generator vllm --model meta-llama/Llama-3-8B --max-samples 50
  
  # 评估 RULER（指定数据集名称）
  python -m eval.run --dataset ruler --data-path data/ruler_niah.jsonl \\
      --dataset-name niah_single_1 --retriever bm25
        """
    )
    
    # 数据集参数
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["hotpotqa", "narrativeqa", "locomo", "ruler"],
        help="数据集名称"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="数据集路径（文件或目录）"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="",
        help="数据集子集名称（仅 RULER 需要）"
    )
    
    # Generator 参数
    parser.add_argument(
        "--generator",
        type=str,
        default="openai",
        choices=["openai", "vllm"],
        help="生成器类型"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="模型名称或路径"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API Key（或从环境变量 OPENAI_API_KEY 读取）"
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="OpenAI API Base URL"
    )
    
    # Retriever 参数
    parser.add_argument(
        "--retriever",
        type=str,
        default="dense",
        choices=["index", "bm25", "dense"],
        help="检索器类型"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="Embedding 模型路径（Dense Retriever）"
    )
    
    # 评估参数
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大样本数（用于快速测试）"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="文本块大小（token 数）"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="检索 top-k 个相关片段"
    )
    
    # 输出参数
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="输出目录"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存预测结果"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式（减少输出）"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 如果没有提供 API Key，尝试从环境变量读取
    if args.generator == "openai" and not args.api_key:
        args.api_key = os.environ.get("OPENAI_API_KEY")
        if not args.api_key:
            print("错误: 使用 OpenAI Generator 需要提供 --api-key 或设置环境变量 OPENAI_API_KEY")
            sys.exit(1)
    
    # 创建配置
    config = BenchmarkConfig(
        data_path=args.data_path,
        generator_type=args.generator,
        model_name=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
        retriever_type=args.retriever,
        embedding_model=args.embedding_model,
        max_samples=args.max_samples,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        output_dir=args.output_dir,
        save_predictions=not args.no_save,
        verbose=not args.quiet,
    )
    
    # 创建对应的 Benchmark
    print(f"\n{'='*60}")
    print(f"GAM Framework - {args.dataset.upper()} 评估")
    print(f"{'='*60}\n")
    
    if args.dataset == "hotpotqa":
        benchmark = HotpotQABenchmark(config)
    elif args.dataset == "narrativeqa":
        benchmark = NarrativeQABenchmark(config)
    elif args.dataset == "locomo":
        benchmark = LoCoMoBenchmark(config)
    elif args.dataset == "ruler":
        benchmark = RULERBenchmark(config, dataset_name=args.dataset_name)
    else:
        print(f"错误: 不支持的数据集 {args.dataset}")
        sys.exit(1)
    
    # 运行评估
    try:
        results = benchmark.run()
        
        # 打印结果
        print(f"\n{'='*60}")
        print("评估结果:")
        print(f"{'='*60}")
        for metric, value in results.items():
            if isinstance(value, float):
                print(f"  {metric:20s}: {value:.4f}")
            else:
                print(f"  {metric:20s}: {value}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n错误: 评估过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


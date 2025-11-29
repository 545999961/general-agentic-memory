#!/bin/bash
# GAM Benchmarks 配置示例
# 复制此文件并根据你的需求修改

# ========================================
# 通用配置
# ========================================

# OpenAI API 配置
export OPENAI_API_KEY="sk-your-api-key-here"
export OPENAI_API_BASE="https://api.openai.com/v1"  # 可选：自定义endpoint

# 数据路径（根据你的实际路径修改）
export DATA_DIR="./data"
export HOTPOTQA_PATH="${DATA_DIR}/hotpotqa.json"
export NARRATIVEQA_PATH="narrativeqa"  # HuggingFace dataset name
export LOCOMO_PATH="${DATA_DIR}/locomo.json"
export RULER_PATH="${DATA_DIR}/ruler.jsonl"

# 输出路径
export OUTPUT_DIR="./outputs"

# ========================================
# 模型配置
# ========================================

# OpenAI 模型
export DEFAULT_GENERATOR="openai"
export DEFAULT_MODEL="gpt-4"
# export DEFAULT_MODEL="gpt-3.5-turbo"  # 更便宜的选择

# VLLM 本地模型（如果使用）
# export DEFAULT_GENERATOR="vllm"
# export DEFAULT_MODEL="/path/to/local/model"

# ========================================
# 检索器配置
# ========================================

# Dense Retriever（推荐）
export DEFAULT_RETRIEVER="dense"
export EMBEDDING_MODEL="BAAI/bge-base-en-v1.5"

# BM25 Retriever（更快但效果可能略差）
# export DEFAULT_RETRIEVER="bm25"

# Index Retriever（最简单）
# export DEFAULT_RETRIEVER="index"

# ========================================
# 评估参数
# ========================================

# 最大样本数（用于快速测试）
# export MAX_SAMPLES=10     # 快速测试
# export MAX_SAMPLES=50     # 中等规模
export MAX_SAMPLES=""       # 全部数据

# 文本块大小（token数）
export CHUNK_SIZE=2000

# 检索数量
export TOP_K=5

# ========================================
# 使用示例
# ========================================

# 加载此配置文件：
#   source scripts/example_config.sh

# 然后运行评估：
#   bash scripts/eval_hotpotqa.sh --data-path $HOTPOTQA_PATH

# 或使用 Python CLI：
#   python -m eval.run \
#       --dataset hotpotqa \
#       --data-path $HOTPOTQA_PATH \
#       --generator $DEFAULT_GENERATOR \
#       --model $DEFAULT_MODEL \
#       --retriever $DEFAULT_RETRIEVER

# ========================================
# 不同场景的配置示例
# ========================================

# 场景 1: 快速测试
quick_test() {
    export MAX_SAMPLES=10
    export DEFAULT_MODEL="gpt-3.5-turbo"
}

# 场景 2: 完整评估
full_eval() {
    export MAX_SAMPLES=""
    export DEFAULT_MODEL="gpt-4"
}

# 场景 3: 本地模型评估
local_eval() {
    export DEFAULT_GENERATOR="vllm"
    export DEFAULT_MODEL="/path/to/local/model"
    export MAX_SAMPLES=50
}

# 场景 4: 低成本评估
low_cost() {
    export DEFAULT_MODEL="gpt-3.5-turbo"
    export DEFAULT_RETRIEVER="bm25"
    export MAX_SAMPLES=100
}

# 使用场景：
#   quick_test
#   bash scripts/eval_hotpotqa.sh --data-path $HOTPOTQA_PATH


#!/bin/bash
# NarrativeQA 评估脚本

set -e

# 默认参数
DATA_PATH="narrativeqa"
GENERATOR="openai"
MODEL="gpt-4"
RETRIEVER="dense"
EMBEDDING_MODEL="BAAI/bge-base-en-v1.5"
MAX_SAMPLES=""
OUTPUT_DIR="outputs/narrativeqa"
API_KEY=""
API_BASE=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --generator)
            GENERATOR="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --retriever)
            RETRIEVER="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="--api-key $2"
            shift 2
            ;;
        --api-base)
            API_BASE="--api-base $2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="--max-samples $2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "NarrativeQA 评估"
echo "=========================================="
echo "数据路径: $DATA_PATH"
echo "生成器: $GENERATOR"
echo "模型: $MODEL"
echo "检索器: $RETRIEVER"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="

python -m eval.run \
    --dataset narrativeqa \
    --data-path "$DATA_PATH" \
    --generator "$GENERATOR" \
    --model "$MODEL" \
    --retriever "$RETRIEVER" \
    --embedding-model "$EMBEDDING_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    $API_KEY \
    $API_BASE \
    $MAX_SAMPLES

echo ""
echo "评估完成！结果已保存到: $OUTPUT_DIR"


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mem0 + RULER 数据集测试文件

结合 GAM 的 VLLM 生成器与 mem0 (FAISS) 记忆系统，对 RULER 数据集进行问答评测。
数据集支持传入单个 JSONL 文件或包含多个 JSONL 的目录。
"""

import os
import glob
import logging
import sys
import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# 二次保险：进程一启动就关
os.environ.setdefault("MEM0_TELEMETRY", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("POSTHOG_DISABLED", "1")
# 静音 backoff 日志（仅不影响功能）
logging.getLogger("backoff").setLevel(logging.ERROR)

# 极端保险：如果 posthog 已被别处间接 import
try:
    import posthog  # type: ignore

    posthog.disabled = True
except Exception:
    pass


# ==============  导入你本地 GAM 的 vLLM 生成器  ==============
# 从 /share/project/bingyu/code/general-agentic-memory/cf/cf_eval/test_mem0_ruler.py
# 到 /share/project/bingyu/code，需要向上4级
code_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)  # /share/project/bingyu/code
gam_path = os.path.join(code_dir, "general-agentic-memory")
if os.path.exists(gam_path):
    sys.path.insert(0, gam_path)
from gam import VLLMGenerator, VLLMGeneratorConfig, OpenAIGenerator, OpenAIGeneratorConfig  # noqa: E402

# ==============  mem0  ==============
from mem0 import Memory  # noqa: E402


# -------------------------
# 数据加载
# -------------------------
def load_ruler_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """
    加载单个 RULER JSONL 文件。
    期望字段：context, question, outputs(List[str])，可选 example/instruction。
    """
    data_all: List[Dict[str, Any]] = []
    dataset_name = os.path.splitext(os.path.basename(jsonl_path))[0]
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            sample_id = f"{dataset_name}-{idx}"
            item.update(
                {
                    "index": item.get("index", idx),
                    "_id": sample_id,
                    "dataset": dataset_name,
                }
            )
            data_all.append(item)
    return data_all


def load_ruler_data(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    支持从单个文件或目录加载 RULER 数据。
    返回以 dataset 名为 key 的字典。
    """
    datasets: Dict[str, List[Dict[str, Any]]] = {}
    jsonl_files: List[str] = []

    if os.path.isfile(path):
        jsonl_files = [path]
    elif os.path.isdir(path):
        jsonl_files = sorted(glob.glob(os.path.join(path, "*.jsonl")))
    else:
        raise FileNotFoundError(f"路径不存在: {path}")

    if not jsonl_files:
        raise FileNotFoundError(f"未在 {path} 找到 JSONL 文件")

    for file in jsonl_files:
        samples = load_ruler_jsonl(file)
        if not samples:
            continue
        dataset = samples[0].get("dataset", os.path.splitext(os.path.basename(file))[0])
        datasets.setdefault(dataset, []).extend(samples)

    for dataset_name in datasets:
        datasets[dataset_name].sort(key=lambda x: x.get("index", 0))

    return datasets


# -------------------------
# 长文本切分（以嵌入模型 tokenizer 为准，默认 512）
# -------------------------
def build_context_chunks_for_sample(
    sample: Dict[str, Any],
    max_tokens: int = 512,
    embedding_tokenizer=None,
) -> List[str]:
    context_text = sample.get("context") or ""
    if not context_text:
        return []

    if embedding_tokenizer is not None:
        try:
            tokens = embedding_tokenizer.encode(context_text, add_special_tokens=False)
            if len(tokens) <= max_tokens:
                return [f"[Session 1]\n{context_text}"]
            return _smart_split_by_tokens(context_text, tokens, max_tokens, embedding_tokenizer)
        except Exception as e:
            print(f"Warning: embedding tokenizer split failed: {e}. Falling back to tiktoken/char.")

    try:
        import tiktoken

        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
        tokens = tokenizer.encode(context_text, disallowed_special=())
        if len(tokens) <= max_tokens:
            return [f"[Session 1]\n{context_text}"]
        return _smart_split_by_tokens(context_text, tokens, max_tokens, tokenizer)
    except Exception:
        return _fallback_char_split(context_text, max_tokens)


def _smart_split_by_tokens(text: str, tokens: List[int], max_tokens: int, tokenizer) -> List[str]:
    chunks: List[str] = []
    sid, start = 1, 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        try:
            chunk_text = tokenizer.decode(tokens[start:end], skip_special_tokens=True)
        except TypeError:
            chunk_text = tokenizer.decode(tokens[start:end])
        chunk_text = chunk_text.strip()
        if chunk_text:
            chunks.append(f"[Session {sid}]\n{chunk_text}")
            sid += 1
        start = end
    return chunks


def _fallback_char_split(text: str, max_tokens: int) -> List[str]:
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return [f"[Session 1]\n{text}"]
    chunks: List[str] = []
    cur, sid = 0, 1
    while cur < len(text):
        end = min(cur + max_chars, len(text))
        if end < len(text):
            nl = text.rfind("\n", cur, end)
            if nl > cur:
                end = nl
            else:
                sp = text.rfind(" ", cur, end)
                if sp > cur:
                    end = sp
        part = text[cur:end].strip()
        if part:
            chunks.append(f"[Session {sid}]\n{part}")
            sid += 1
        cur = end
    return chunks


# -------------------------
# Prompt
# -------------------------
def build_question_prompt(sample: Dict[str, Any]) -> str:
    parts: List[str] = []

    instruction = (sample.get("instruction") or "").strip()
    if instruction:
        parts.append("Instruction:\n" + instruction)

    example = (sample.get("example") or "").strip()
    if example:
        parts.append("Here is an example:\n" + example)

    question = (sample.get("question") or "").strip()
    if question:
        parts.append("Question:\n" + question)

    return "\n\n".join(parts).strip()


def make_prompt(summary: str, question_prompt: str) -> str:
    prompt = f"""Read the text below and answer a question. Context: {summary}\n\n{question_prompt}\n\nAnswer:"""
    return prompt


# -------------------------
# 评测
# -------------------------
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def evaluate_answer(model_response: str, ground_truth_outputs: List[str]) -> bool:
    if not ground_truth_outputs or not model_response:
        return False

    model_response_lower = model_response.lower()
    model_response_normalized = normalize_text(model_response)

    unique_answers = list(dict.fromkeys(ground_truth_outputs))

    for answer in unique_answers:
        answer_str = str(answer).strip()
        if not answer_str:
            continue

        answer_lower = answer_str.lower()
        if answer_lower in model_response_lower:
            continue

        answer_normalized = normalize_text(answer_str)
        if answer_normalized in model_response_normalized:
            continue

        answer_words = [w for w in answer_normalized.split() if len(w) > 2]
        if answer_words and all(word in model_response_normalized for word in answer_words):
            continue

        return False

    return True


# -------------------------
# mem0 初始化（FAISS 目录 + 归一化 + CPU embedder + reranker）
# -------------------------
# def init_mem0_faiss(
#     index_dir: str,
#     vllm_url: str = "http://localhost:8000/v1",
#     vllm_model: str = "qwen2.5-14b-instruct",
#     hf_embed_model: str = "/share/project/bingyu/models/bge-base-en-v1.5",
#     embed_dims: int = 768,
#     custom_fact_prompt: Optional[str] = None,
#     metric: str = "cosine",
# ) -> Memory:\
def init_mem0_faiss(
    index_dir: str,
    api_base_url: str,
    api_model: str,
    api_key: str,
    hf_embed_model: str = "/share/project/bingyu/models/bge-base-en-v1.5",
    embed_dims: int = 768,
    custom_fact_prompt: Optional[str] = None,
    metric: str = "cosine",
) -> Memory:
    os.makedirs(index_dir, exist_ok=True)
    dist = {"cosine": "cosine", "ip": "inner_product", "l2": "euclidean"}.get(metric, "cosine")

    cfg = {
        # "version": "v1.1",
        # "llm": {
        #     "provider": "vllm",
        #     "config": {"model": vllm_model, "vllm_base_url": vllm_url, "temperature": 0.3, "max_tokens": 512},
        # },
        "version": "v1.1",
        "llm": {
            "provider": "openai",
            "config": {
                "model": api_model,
                "api_key": api_key,
                "openai_base_url": api_base_url,
                "temperature": 0.3,
                "max_tokens": 512,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": hf_embed_model,
                "embedding_dims": embed_dims,
            },
        },
        "vector_store": {
            "provider": "faiss",
            "config": {
                "collection_name": "ruler",
                "path": index_dir,
                "embedding_model_dims": embed_dims,
                "distance_strategy": dist,
                "normalize_L2": True,
            },
        },
    }
    if custom_fact_prompt:
        cfg["custom_fact_extraction_prompt"] = custom_fact_prompt

    print(f"[mem0] embedder={hf_embed_model} dims={embed_dims} faiss_dir={index_dir}")
    return Memory.from_config(config_dict=cfg)


# -------------------------
# 写入 & 检索
# -------------------------
def mem0_add_context_chunks(m: Memory, sample_id: str, chunks: List[str]) -> int:
    added_count = 0
    for i, chunk in enumerate(chunks, 1):
        try:
            m.add(chunk, user_id=sample_id, metadata={"sample_id": sample_id, "session": i}, infer=False)
            added_count += 1
        except Exception as e:
            print(f"[Warning] 写入记忆失败 (chunk {i}): {e}")
    print(f"[DEBUG] 成功写入 {added_count}/{len(chunks)} 个记忆块")
    return added_count


def mem0_search_relevant(m: Memory, sample_id: str, query: str, limit: int = 20) -> Tuple[str, int]:
    try:
        out = m.search(query, user_id=sample_id, limit=limit)
        print(f"[DEBUG] search 返回类型: {type(out)}")
        print(f"[DEBUG] search 返回内容: {out}")

        results = out.get("results", []) if isinstance(out, dict) else (out or [])
        mem_texts = []
        for r in results:
            if isinstance(r, dict):
                mem_texts.append(r.get("memory") or r.get("content") or r.get("text") or str(r))
            else:
                mem_texts.append(str(r))
        return "\n".join(f"- {t}" for t in mem_texts), len(mem_texts)
    except Exception as e:
        print(f"[Error] 检索失败: {e}")
        return "", 0


def mem0_clear_user_memories(m: Memory, sample_id: str) -> int:
    try:
        all_memories = m.get_all(user_id=sample_id)
        if not all_memories:
            print(f"[DEBUG] 用户 {sample_id} 没有记忆需要清空")
            return 0

        deleted_count = 0
        memories_list = all_memories.get("results", []) if isinstance(all_memories, dict) else (all_memories or [])

        for mem in memories_list:
            try:
                mem_id = mem.get("id") if isinstance(mem, dict) else None
                if mem_id:
                    m.delete(mem_id)
                    deleted_count += 1
            except Exception as e:
                print(f"[WARNING] 删除记忆失败: {e}")

        print(f"[DEBUG] 已删除 {deleted_count} 条记忆（用户: {sample_id}）")
        return deleted_count
    except Exception as e:
        print(f"[WARNING] 清空记忆失败: {e}")
        return 0


# -------------------------
# 单样本流程
# -------------------------
# def process_sample_with_mem0(
#     sample: Dict[str, Any],
#     sample_index: int,
#     outdir: str,
#     generator: VLLMGenerator,
#     max_tokens: int = 512,
#     embedding_tokenizer=None,
#     mem_limit: int = 20,
#     clear_after_sample: bool = False,
#     vllm_url: str = "http://localhost:8000/v1",
#     vllm_model: str = "qwen2.5-14b-instruct",
#     hf_embed_model: str = "/share/project/bingyu/models/bge-base-en-v1.5",
#     embed_dims: int = 768,
#     custom_fact_prompt: Optional[str] = None,
#     metric: str = "cosine",
# ) -> Dict[str, Any]:
def process_sample_with_mem0(
    sample: Dict[str, Any],
    sample_index: int,
    outdir: str,
    generator,
    max_tokens: int = 512,
    embedding_tokenizer=None,
    mem_limit: int = 20,
    clear_after_sample: bool = False,
    api_base_url: str = "https://api.key77qiqi.com/v1",
    api_model: str = "gpt-4o-mini",
    api_key: str = "sk-2nFJX3Ttm2DSNC5DJpgl8t8M5TrqeqpWBg2NvFwVHhP9Cg5u",
    hf_embed_model: str = "/share/project/bingyu/models/bge-base-en-v1.5",
    embed_dims: int = 768,
    custom_fact_prompt: Optional[str] = None,
    metric: str = "cosine",
) -> Dict[str, Any]:
    sample_id = sample.get("_id", f"sample-{sample_index}")
    dataset_name = sample.get("dataset", "unknown")

    print("\n" + "=" * 60)
    print(f"处理样本 #{sample_index}: {sample_id} (dataset={dataset_name})")
    print("=" * 60)

    chunks = build_context_chunks_for_sample(sample, max_tokens, embedding_tokenizer)
    print(f"上下文块数: {len(chunks)}")
    if chunks:
        print(f"首块预览:\n{chunks[0][:400]}...")

    sample_dir = os.path.join(outdir, sample_id)
    os.makedirs(sample_dir, exist_ok=True)

    index_dir = os.path.join(sample_dir, "faiss_index")
    print("\n步骤 0: 初始化 mem0 Memory")
    t_init = time.time()
    # m = init_mem0_faiss(
    #     index_dir=index_dir,
    #     vllm_url=vllm_url,
    #     vllm_model=vllm_model,
    #     hf_embed_model=hf_embed_model,
    #     embed_dims=embed_dims,
    #     custom_fact_prompt=custom_fact_prompt,
    #     metric=metric,
    # )

    m = init_mem0_faiss(
        index_dir=index_dir,
        api_base_url=api_base_url,
        api_model=api_model,
        api_key=api_key,
        hf_embed_model=hf_embed_model,
        embed_dims=embed_dims,
        custom_fact_prompt=custom_fact_prompt,
        metric=metric,
    )
    print(f"[OK] Memory 初始化完成")
    print(f"[TIME] init={time.time()-t_init:.2f}s")

    print("\n步骤 1: 写入 mem0 记忆")
    t0 = time.time()
    added_count = mem0_add_context_chunks(m, sample_id, chunks)
    print(f"[OK] 记忆写入完成: {added_count}/{len(chunks)}")
    print(f"[TIME] add={time.time()-t0:.2f}s")

    question = sample.get("question", "")
    ground_truth = sample.get("outputs", [])

    print("\n步骤 2: mem0 检索")
    t1 = time.time()
    retrieved_str, retrieved_count = mem0_search_relevant(m, sample_id, question, limit=mem_limit)
    print(f"[OK] 检索到 {retrieved_count} 条相关记忆")
    print(f"[TIME] search={time.time()-t1:.2f}s")

    research_summary = retrieved_str if retrieved_str else "\n".join(chunks[:2])
    question_prompt = build_question_prompt(sample) or question

    trace_payload = {
        "question": question,
        "question_prompt": question_prompt,
        "retrieved_count": retrieved_count,
        "retrieved_memories": retrieved_str,
    }
    with open(os.path.join(sample_dir, "mem0_trace.json"), "w", encoding="utf-8") as f:
        json.dump(trace_payload, f, ensure_ascii=False, indent=2)

    print("\n步骤 3: 生成答案")
    t2 = time.time()
    generator.max_tokens = 256

    attempts, answer_text = 0, ""
    while attempts <= 10:
        try:
            prompt = make_prompt(research_summary, question_prompt)
            resp = generator.generate_single(prompt=prompt)
            answer_text = (resp or {}).get("text", "").strip()
            break
        except Exception as e:
            attempts += 1
            if attempts > 10:
                print(f"[ERROR] 生成答案失败（重试10次后）: {e}")
                answer_text = ""
    print(f"[TIME] gen={time.time()-t2:.2f}s")

    pred = answer_text
    is_correct = evaluate_answer(pred, ground_truth)
    accuracy = 1.0 if is_correct else 0.0

    result = {
        "_id": sample_id,
        "index": sample.get("index", sample_index),
        "dataset": dataset_name,
        "question": question,
        "question_prompt": question_prompt,
        "example": sample.get("example", ""),
        "instruction": sample.get("instruction", ""),
        "outputs": ground_truth,
        "response": answer_text,
        "pred": pred,
        "is_correct": is_correct,
        "accuracy": accuracy,
        "retrieved_count": retrieved_count,
        "num_context_chunks": len(chunks),
        "sample_dir": sample_dir,
    }
    with open(os.path.join(sample_dir, "qa_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n预测答案: {pred}")
    print(f"标准答案: {ground_truth}")
    print(f"是否正确: {is_correct}")
    print(f"[OK] 结果保存到: {os.path.join(sample_dir, 'qa_result.json')}")

    if clear_after_sample:
        print("\n步骤 4: 清空当前 sample 的记忆")
        t3 = time.time()
        deleted = mem0_clear_user_memories(m, sample_id)
        print(f"[OK] 清空完成，删除 {deleted} 条记忆")
        print(f"[TIME] clear={time.time()-t3:.2f}s")
    else:
        print("\n[INFO] 跳过清空记忆（保留用于后续分析）")

    result["status"] = "success"

    del m
    return result


# -------------------------
# 主函数
# -------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="GAM + mem0（FAISS）+ RULER")
    parser.add_argument(
        "--data",
        type=str,
        default="/share/project/bingyu/datasets/ruler/128k/data_jsonl",
        help="RULER 数据集 JSONL 文件或目录路径",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./runs/ruler_mem0_faiss",
        help="输出目录",
    )
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--thread-count", type=int, default=1)

    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--embedding-model-path",
        type=str,
        default="/share/project/bingyu/models/bge-base-en-v1.5",
    )
    parser.add_argument("--mem-limit", type=int, default=20)

    parser.add_argument(
        "--api-base-url",
        type=str,
        default="https://api.key77qiqi.com/v1",
        help="OpenAI 兼容 API base_url，例如 https://api.key77qiqi.com/v1",
    )
    parser.add_argument(
        "--api-model",
        type=str,
        default="gpt-4o-mini",
        help="用于回答 RULER 问题的模型名",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="sk-2nFJX3Ttm2DSNC5DJpgl8t8M5TrqeqpWBg2NvFwVHhP9Cg5u",
        help="OpenAI 兼容 API 的 key（为空则从环境变量 OPENAI_API_KEY 读取）",
    )

    parser.add_argument(
        "--hf-embed-model",
        type=str,
        default="/share/project/bingyu/models/bge-base-en-v1.5",
    )
    parser.add_argument("--embed-dims", type=int, default=768)
    parser.add_argument("--faiss-metric", type=str, default="cosine", choices=["cosine", "ip", "l2"])

    parser.add_argument("--use-custom-fact-prompt", action="store_true")
    parser.add_argument("--sample-timeout-sec", type=int, default=600)
    parser.add_argument(
        "--clear-after-sample",
        action="store_true",
        help="处理完每个 sample 后清空其记忆",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GAM + mem0（FAISS 本地索引）+ RULER")
    print("=" * 60)
    print(f"数据: {args.data}")
    print(f"输出目录: {args.outdir}")
    print(f"LLM: {args.api_model} @ {args.api_base_url}")
    print(f"清空记忆: {'是' if args.clear_after_sample else '否'} (--clear-after-sample)")
    print("=" * 60)

    datasets = load_ruler_data(args.data)
    if not datasets:
        print("没有加载到任何样本")
        return

    custom_fact_prompt: Optional[str] = None
    if args.use_custom_fact_prompt:
        custom_fact_prompt = (
            "Extract only atomic, query-usable facts helpful for multi-hop QA "
            "(entities, aliases, bridge facts, dates, numbers, relations). "
            "One fact per line; no summaries."
        )
    print('test')
    os.makedirs(args.outdir, exist_ok=True)

    # gen_cfg = VLLMGeneratorConfig(
    #     model_name=args.vllm_model,
    #     api_key="empty",
    #     base_url=args.vllm_url,
    #     temperature=0.3,
    #     max_tokens=4096,
    # )
    # generator = VLLMGenerator(gen_cfg.__dict__)

    gen_cfg = OpenAIGeneratorConfig(
        model_name="gpt-4o-mini",
        api_key="sk-2nFJX3Ttm2DSNC5DJpgl8t8M5TrqeqpWBg2NvFwVHhP9Cg5u",
        base_url="https://api.key77qiqi.com/v1",
        temperature=0.3,
        max_tokens=512
    )
    generator = OpenAIGenerator(gen_cfg.__dict__)   


    shared_tokenizer = None
    if args.embedding_model_path:
        try:
            from transformers import AutoTokenizer

            print(f"[INFO] 加载共享 tokenizer: {args.embedding_model_path}")
            shared_tokenizer = AutoTokenizer.from_pretrained(
                args.embedding_model_path,
                local_files_only=True,
                use_fast=True,
            )
            print("[OK] 共享 tokenizer 加载完成\n")
        except Exception as e:
            print(f"[WARNING] tokenizer 加载失败: {e}，将使用备用切分方法\n")

    overall_results: List[Dict[str, Any]] = []
    overall_dataset_stats: Dict[str, Dict[str, Any]] = {}

    for dataset_name, samples in datasets.items():
        print(f"\n{'=' * 80}")
        print(f"处理数据集: {dataset_name}")
        print(f"{'=' * 80}")
        print(f"共加载 {len(samples)} 个样本")

        start_idx = max(args.start_idx, 0)
        end_idx = args.end_idx if args.end_idx is not None else len(samples)
        end_idx = min(end_idx, len(samples))
        if start_idx >= end_idx:
            print(f"[WARNING] 样本范围无效，跳过数据集 {dataset_name}")
            continue

        dataset_outdir = os.path.join(args.outdir, dataset_name)
        os.makedirs(dataset_outdir, exist_ok=True)

        indices = list(range(start_idx, end_idx))
        dataset_results: List[Dict[str, Any]] = []

        for idx in tqdm(indices, desc=f"处理 {dataset_name}"):
            sample = samples[idx]
            sample_index = sample.get("index", idx)
            try:
                # result = process_sample_with_mem0(
                #     sample=sample,
                #     sample_index=sample_index,
                #     outdir=dataset_outdir,
                #     generator=generator,
                #     max_tokens=args.max_tokens,
                #     embedding_tokenizer=shared_tokenizer,
                #     mem_limit=args.mem_limit,
                #     clear_after_sample=args.clear_after_sample,
                #     vllm_url=args.vllm_url,
                #     vllm_model=args.vllm_model,
                #     hf_embed_model=args.hf_embed_model,
                #     embed_dims=args.embed_dims,
                #     custom_fact_prompt=custom_fact_prompt,
                #     metric=args.faiss_metric,
                # )
                result = process_sample_with_mem0(
                    sample=sample,
                    sample_index=sample_index,
                    outdir=dataset_outdir,
                    generator=generator,
                    max_tokens=args.max_tokens,
                    embedding_tokenizer=shared_tokenizer,
                    mem_limit=args.mem_limit,
                    clear_after_sample=args.clear_after_sample,
                    api_base_url=args.api_base_url,
                    api_model=args.api_model,
                    api_key=args.api_key,
                    hf_embed_model=args.hf_embed_model,
                    embed_dims=args.embed_dims,
                    custom_fact_prompt=custom_fact_prompt,
                    metric=args.faiss_metric,
                )
                dataset_results.append(result)
                import gc

                gc.collect()
            except Exception as e:
                print(f"[ERROR] sample #{sample_index} ({sample.get('_id', sample_index)}) 失败: {e}")
                import traceback

                traceback.print_exc()
                dataset_results.append(
                    {
                        "_id": sample.get("_id", f"sample-{sample_index}"),
                        "index": sample_index,
                        "dataset": dataset_name,
                        "status": "error",
                        "error": str(e),
                    }
                )

        overall_results.extend(dataset_results)

        success_results = [r for r in dataset_results if r.get("status") == "success"]
        success_count = len(success_results)
        correct_count = sum(1 for r in success_results if r.get("is_correct"))
        accuracy = (correct_count / success_count) if success_count else 0.0
        error_count = len(dataset_results) - success_count

        print(f"\n{'-' * 60}")
        print(f"{dataset_name} 数据集统计")
        print(f"{'-' * 60}")
        print(f"处理样本数: {len(dataset_results)}")
        print(f"成功: {success_count}")
        print(f"错误: {error_count}")
        print(f"正确数: {correct_count}")
        print(f"准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"{'-' * 60}")

        summary_file = os.path.join(dataset_outdir, f"batch_results_{start_idx}_{end_idx-1}.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(dataset_results, f, ensure_ascii=False, indent=2)
        print(f"[OK] 结果汇总保存: {summary_file}")

        stats_payload = {
            "dataset": dataset_name,
            "start_idx": start_idx,
            "end_idx": end_idx - 1,
            "total": len(dataset_results),
            "success": success_count,
            "errors": error_count,
            "correct": correct_count,
            "accuracy": accuracy,
        }
        stats_file = os.path.join(dataset_outdir, f"batch_statistics_{start_idx}_{end_idx-1}.json")
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats_payload, f, ensure_ascii=False, indent=2)
        print(f"[OK] 统计信息保存: {stats_file}")

        overall_dataset_stats[dataset_name] = stats_payload

    overall_success = [r for r in overall_results if r.get("status") == "success"]
    overall_errors = len(overall_results) - len(overall_success)
    overall_correct = sum(1 for r in overall_success if r.get("is_correct"))
    overall_accuracy = (overall_correct / len(overall_success)) if overall_success else 0.0

    overall_summary = {
        "total_samples": len(overall_results),
        "success_samples": len(overall_success),
        "error_samples": overall_errors,
        "correct_samples": overall_correct,
        "overall_accuracy": overall_accuracy,
        "dataset_stats": overall_dataset_stats,
    }
    overall_summary_file = os.path.join(
        args.outdir,
        f"overall_summary_{args.start_idx}_{args.end_idx if args.end_idx is not None else 'all'}.json",
    )
    with open(overall_summary_file, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 总体汇总保存: {overall_summary_file}")
    print(f"成功样本: {len(overall_success)} | 错误样本: {overall_errors}")
    print(f"总体准确率: {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)")
    print(f"\n处理完成！结果保存在: {args.outdir}")



if __name__ == "__main__":
    main()


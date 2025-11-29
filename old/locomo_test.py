#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAM 框架 + LoCoMo 数据集测试文件

结合 locomoqa_v3.py 的数据处理逻辑和 GAM 框架，测试在多轮对话数据上的效果。
"""

import sys
import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from tqdm import tqdm

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from gam import (
    MemoryAgent,
    ResearchAgent,
    VLLMGenerator,
    InMemoryMemoryStore,
    InMemoryPageStore,
    IndexRetriever,
    BM25Retriever,
    DenseRetriever,
    VLLMGeneratorConfig,
    OpenAIGenerator,
    OpenAIGeneratorConfig,
    IndexRetrieverConfig,
    BM25RetrieverConfig,
    DenseRetrieverConfig,
)

# ========== 数据加载：借鉴自 locomoqa_v3.py ==========

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_locomo(json_path: str) -> List[Dict[str, Any]]:
    """Load LoCoMo JSON and return the list of samples."""
    data = load_json(json_path)
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    if isinstance(data, list):
        return data
    raise ValueError("Unrecognized LoCoMo JSON shape. Expect a list or {'samples': [...]}.")

def extract_sessions(conv_obj: Dict[str, Any]) -> List[Tuple[int, str, List[Dict[str, Any]], Optional[str]]]:
    """
    Extract sessions as (idx, timestamp, turns, optional_session_summary).
    """
    sessions: List[Tuple[int, str, List[Dict[str, Any]], Optional[str]]] = []
    for k, v in conv_obj.items():
        m = re.match(r'^session_(\d+)$', k)
        if not (m and isinstance(v, list)):
            continue
        idx = int(m.group(1))
        ts = conv_obj.get(f"session_{idx}_date_time", "")
        ssum = conv_obj.get(f"session_{idx}_summary", None)
        sessions.append((idx, ts, v, ssum if isinstance(ssum, str) and ssum.strip() else None))
    sessions.sort(key=lambda x: x[0])
    return sessions

def session_to_text(idx: int, ts: str, turns: List[Dict[str, Any]], session_summary: Optional[str]) -> str:
    # 将时间信息放在最前面，使用更突出的格式
    lines = [f"=== SESSION {idx} - Dialogue Time(available to answer questions): {ts} ==="]
    lines.append("")  # 空行分隔
    
    for turn in turns:
        speaker = turn.get("speaker", "Unknown")
        dia_id  = turn.get("dia_id", "")
        text    = turn.get("text", "")
        lines.append(f"{speaker} ({dia_id}): {text}")
    
    if session_summary:
        lines.append("")
        lines.append(f"Session {idx} summary: {session_summary}")
    
    return "\n".join(lines).strip()

def build_session_chunks_for_sample(sample: Dict[str, Any]) -> List[str]:
    """Build session chunks from a sample."""
    conv = sample.get("conversation", {})
    sessions = extract_sessions(conv)
    chunks: List[str] = []
    for idx, ts, turns, ssum in sessions:
        chunks.append(session_to_text(idx, ts, turns, ssum))
    return chunks

def collect_qa_items_for_sample(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect QA items from a sample."""
    qas: List[Dict[str, Any]] = []
    sid = sample.get("sample_id", None)
    for q in sample.get("qa", []):
        qas.append({
            "sample_id": sid,
            "question": q.get("question"),
            "answer": q.get("answer"),
            "category": q.get("category"),
            "evidence": q.get("evidence"),
        })
    return qas

# ========== Prompt 设计：完全借鉴自 locomoqa_v3.py ==========

def safe_json_extract(candidate: Any) -> Optional[Dict[str, Any]]:
    """尽量把模型输出（string/dict）解析成 dict，失败返回 None。"""
    if isinstance(candidate, dict):
        return candidate
    if not isinstance(candidate, str):
        return None
    s = candidate.strip()
    l = s.find('{')
    r = s.rfind('}')
    if l == -1 or r == -1 or r <= l:
        return None
    try:
        return json.loads(s[l:r+1])
    except Exception:
        return None

def make_summary_prompt(summary: str, question: str) -> str:
    return f"""
    Based on the summary below, write an answer in the form of **a short phrase** for the following question, not a sentence. Answer with exact words from the context whenever possible.
    For questions that require answering a date or time, strictly follow the format \"15 July 2023\" and provide a specific date whenever possible. For example, if you need to answer \"last year,\" give the specific year of last year rather than just saying \"last year.\" Only provide one year, date, or time, without any extra responses.
    If the question is about the duration, answer in the form of several years, months, or days.
   
    QUESTION:
    {question}

    SUMMARY:
    {summary}

    Short answer:
    """

def make_summary_prompt_category3(summary: str, question: str) -> str:
    return f"""
    Based on the summary below, write an answer in the form of **a short phrase** for the following question, not a sentence.
    The question may need you to analyze and infer the answer from the summary.
     
    QUESTION:
    {question}

    SUMMARY:
    {summary}

    Short answer:
    """

def make_memory_only_prompt(memory_obj: Any, question: str) -> str:
    mem_str = json.dumps(memory_obj, ensure_ascii=False, indent=2) if isinstance(memory_obj, dict) else str(memory_obj)
    return f"""
    Based on the MEMORY STATE below,  write an answer in the form of a brief short phrase for the following question. Answer with exact words from the context whenever possible.
    The date should be written as an exact date.

    MEMORY STATE:
    {mem_str}

    QUESTION:
    {question}

    Short answer:
    """

def make_memory_only_prompt_category3(memory_obj: Any, question: str) -> str:
    mem_str = json.dumps(memory_obj, ensure_ascii=False, indent=2) if isinstance(memory_obj, dict) else str(memory_obj)
    return f"""
    Based on the MEMORY STATE below,  write an answer in the form of a brief short phrase for the following question. Answer with exact words from the context whenever possible.
    The date should be written as an exact date.

    MEMORY STATE:
    {mem_str}

    QUESTION:
    {question}

    Short answer:
    """

def answer_with_summary(category: Optional[int], summary: str, question: str, generator) -> str:
    """根据category选择不同的prompt"""
    if category == 3:
        prompt = make_summary_prompt_category3(summary, question)
    else:
        prompt = make_summary_prompt(summary, question)
    raw = generator.generate_single(prompt=prompt)
    return raw.get("text", "").strip()

def answer_with_memory(category: Optional[int], final_memory: Dict[str, Any], question: str, generator) -> str:
    """根据category选择不同的prompt"""
    if category == 3:
        prompt = make_memory_only_prompt_category3(final_memory, question)
    else:
        prompt = make_memory_only_prompt(final_memory, question)
    raw = generator.generate_single(prompt=prompt)
    return raw.get("text", "").strip()

# ========== 核心处理逻辑 ==========

def process_sample(sample: Dict[str, Any], sample_index: int, outdir: str, memory_model_api: str, thread_count: int = 40):
    """
    使用 GAM 框架处理单个样本。
    
    流程：
    1. 使用 MemoryAgent 构建记忆
    2. 使用 ResearchAgent 进行深度研究
    3. 基于研究结果进行问答
    """
    sample_id = sample.get("sample_id", f"conv-{sample_index}")
    
    print(f"\n{'='*60}")
    print(f"处理样本 #{sample_index}: {sample_id}")
    print(f"{'='*60}")
    
    try:
        # 1. 构建会话块
        session_chunks = build_session_chunks_for_sample(sample)
        print(f"会话数: {len(session_chunks)}")
        if session_chunks:
            print(f"第一个会话预览:\n{session_chunks[0][:400]}...")
        
        # 创建输出目录
        sample_results_dir = os.path.join(outdir, sample_id)
        os.makedirs(sample_results_dir, exist_ok=True)
        print(f"输出目录: {sample_results_dir}")
        
        # 2. 创建共享存储
        memory_store = InMemoryMemoryStore(dir_path=sample_results_dir)
        page_store = InMemoryPageStore(dir_path=sample_results_dir)
        
        # 3. 创建 Generator
        print(f"\n步骤 1: 创建 Generator")
        # memory_generator_config = VLLMGeneratorConfig(
        #     model_name="qwen3",
        #     api_key="empty",
        #     base_url=memory_model_api,
        #     temperature=0.3,
        #     max_tokens=2048
        # )
        # memory_generator = VLLMGenerator(memory_generator_config.__dict__)


        memory_generator_config = OpenAIGeneratorConfig(
            model_name="gpt-4o-mini",
            api_key="sk-2nFJX3Ttm2DSNC5DJpgl8t8M5TrqeqpWBg2NvFwVHhP9Cg5u",
            base_url="https://api.key77qiqi.com/v1",
            temperature=0.3,
            max_tokens=256
        )
        memory_generator = OpenAIGenerator(memory_generator_config.__dict__)     
        print(f"[OK] Generator 创建完成")
        
        # 4. 使用 MemoryAgent 构建记忆（将每个 session 作为一条消息）
        print(f"\n步骤 2: 使用 MemoryAgent 构建记忆")
        memory_agent = MemoryAgent(
            memory_store=memory_store,
            page_store=page_store,
            generator=memory_generator
        )
        
        if not os.path.exists(os.path.join(sample_results_dir, 'memory_state.json')):
            for i, session_chunk in enumerate(session_chunks, 1):
                print(f"  处理会话 {i}/{len(session_chunks)}...")
                memory_update = memory_agent.memorize(session_chunk)
        
        # 查看构建的记忆
        final_state = memory_store.load()
        print(f"[OK] 记忆构建完成！共 {len(final_state.abstracts)} 条记忆摘要")
        
        # 显示记忆摘要
        print("\n📚 记忆摘要:")
        for i, abstract in enumerate(final_state.abstracts, 1):
            print(f"  {i}. {abstract[:100]}...")
        
        # 保存记忆状态
        memory_state_file = os.path.join(sample_results_dir, "memory_state.json")
        with open(memory_state_file, 'w', encoding='utf-8') as f:
            json.dump(final_state.model_dump(), f, ensure_ascii=False, indent=2)
        print(f"[OK] 记忆状态已保存: {memory_state_file}")
        
        # 5. 创建检索器
        print(f"\n步骤 3: 创建检索器")
        retrievers = {}
        
        # 索引检索器
        try:
            page_index_dir = os.path.join(sample_results_dir, "page_index")
            # 如果索引目录已存在，先删除它（避免 "Directory not empty" 错误）
            if os.path.exists(page_index_dir):
                import shutil
                shutil.rmtree(page_index_dir)
                print(f"[INFO] 清理已存在的页面索引目录: {page_index_dir}")
            
            index_config = IndexRetrieverConfig(
                index_dir=page_index_dir
            )
            index_retriever = IndexRetriever(index_config.__dict__)
            index_retriever.build(page_store)
            retrievers["page_index"] = index_retriever
            print(f"[OK] 索引检索器创建成功")
        except Exception as e:
            print(f"[WARN] 索引检索器创建失败: {e}")
        
        # BM25 检索器
        try:
            bm25_index_dir = os.path.join(sample_results_dir, "bm25_index")
            # 如果索引目录已存在，先删除它（避免 "Directory not empty" 错误）
            if os.path.exists(bm25_index_dir):
                import shutil
                shutil.rmtree(bm25_index_dir)
                print(f"[INFO] 清理已存在的 BM25 索引目录: {bm25_index_dir}")
            
            bm25_config = BM25RetrieverConfig(
                index_dir=bm25_index_dir,
                threads=1
            )
            bm25_retriever = BM25Retriever(bm25_config.__dict__)
            bm25_retriever.build(page_store)
            retrievers["keyword"] = bm25_retriever
            print(f"[OK] BM25 检索器创建成功")
        except Exception as e:
            print(f"[WARN] BM25 检索器创建失败: {e}")
        
        # Dense 检索器
        try:
            dense_index_dir = os.path.join(sample_results_dir, "dense_index")
            # 如果索引目录已存在，先删除它（避免 "Directory not empty" 错误）
            if os.path.exists(dense_index_dir):
                import shutil
                shutil.rmtree(dense_index_dir)
                print(f"[INFO] 清理已存在的 Dense 索引目录: {dense_index_dir}")
            
            dense_config = DenseRetrieverConfig(
                index_dir=dense_index_dir,
                api_url="http://localhost:8001"  # API 模式：所有进程共享一个模型服务
            )
            dense_retriever = DenseRetriever(dense_config.__dict__)
            dense_retriever.build(page_store)
            retrievers["vector"] = dense_retriever
            print(f"[OK] Dense 检索器创建成功")
        except Exception as e:
            print(f"[WARN] Dense 检索器创建失败: {e}")
        
        print(f"[INFO] 成功创建 {len(retrievers)} 个检索器")
        
        print(f"\n步骤 1: 创建 Generator")
        # generator_config = VLLMGeneratorConfig(
        #     model_name="qwen2.5-14b-instruct",
        #     api_key="empty",
        #     base_url="http://localhost:8000/v1",
        #     temperature=0.3,
        #     max_tokens=2048
        # )
        # generator = VLLMGenerator(generator_config.__dict__)


        # working_config = VLLMGeneratorConfig(
        #     model_name="qwen2.5-14b-instruct",
        #     api_key="empty",
        #     base_url="http://localhost:8000/v1",
        #     temperature=0.3,
        #     max_tokens=64
        # )
        # working_generator = VLLMGenerator(working_config.__dict__)

        generator_config = OpenAIGeneratorConfig(
            model_name="gpt-4o-mini",
            api_key="sk-2nFJX3Ttm2DSNC5DJpgl8t8M5TrqeqpWBg2NvFwVHhP9Cg5u",
            base_url="https://api.key77qiqi.com/v1",
            temperature=0.3,
            max_tokens=2048
        )
        generator = OpenAIGenerator(generator_config.__dict__)     
    

        working_config = OpenAIGeneratorConfig(
            model_name="gpt-4o-mini",
            api_key="sk-2nFJX3Ttm2DSNC5DJpgl8t8M5TrqeqpWBg2NvFwVHhP9Cg5u",
            base_url="https://api.key77qiqi.com/v1",
            temperature=0.3,
            max_tokens=64
        )
        working_generator = OpenAIGenerator(working_config.__dict__)    


        print(f"[OK] Generator 创建完成")





        # system_prompts = {
        #     "planning": "",
        #     "integration": "Integrate the information from the context with exact words from the context whenever possible",
        #     "reflection": ""
        # }


        # 6. 创建 ResearchAgent
        print(f"\n步骤 4: 创建 ResearchAgent")
        research_agent = ResearchAgent(
            page_store=page_store,
            memory_store=memory_store,
            retrievers=retrievers,
            generator=generator,
            # system_prompts=system_prompts,
            max_iters=3
        )
        print(f"[OK] ResearchAgent 创建完成")
        
        # 7. 进行问答（并行处理问题）
        print(f"\n步骤 5: 进行问答")
        qas = collect_qa_items_for_sample(sample)
        print(f"共有 {len(qas)} 个问题需要回答")
        
        # 将记忆转换为字符串格式
        final_memory_str = json.dumps(final_state.model_dump(), ensure_ascii=False, indent=2)
        
        # 定义处理单个问题的worker函数
        def process_question(qi_with_index):
            """处理单个问题的worker函数"""
            i, qi = qi_with_index
            q = qi.get("question") or ""
            gold = qi.get("answer")
            cat = qi.get("category")
            
            print(f"\n--- 问题 {i}/{len(qas)} ---")
            print(f"问题: {q}")
            print(f"标准答案: {gold}")
            print(f"分类: {cat}")
            
            if cat == 5:
                return None

            try:
                # 使用 ResearchAgent 进行研究
                print(f"[问题 {i}] 正在进行深度研究...")
                result = research_agent.research(q)
                research_summary = result.integrated_memory
                print(f"[问题 {i}] [OK] 研究完成！迭代次数: {len(result.raw_memory.get('iterations', []))}")
                print(f"[问题 {i}] 研究摘要: {research_summary[:200]}...")
                
                # 保存研究轨迹
                research_trace = {
                    "question": q,
                    "raw_memory": result.raw_memory,
                    "integrated_memory": result.integrated_memory,
                    "iterations": result.raw_memory.get("iterations", []),
                    "search_plans": result.raw_memory.get("search_plans", []),
                    "reflections": result.raw_memory.get("reflections", [])
                }
                
                # 保存单个问题的研究轨迹
                trace_file = os.path.join(sample_results_dir, f"research_trace_q{i}.json")
                with open(trace_file, 'w', encoding='utf-8') as f:
                    json.dump(research_trace, f, ensure_ascii=False, indent=2)
                print(f"[问题 {i}] [INFO] 研究轨迹已保存: {trace_file}")
                
                # 基于研究结果生成答案（根据category选择不同prompt）
                print(f"[问题 {i}] 生成答案...")
                summary_answer = answer_with_summary(cat, research_summary, q, working_generator)
                memory_answer = answer_with_memory(cat, final_memory_str, q, working_generator)
                
                print(f"[问题 {i}] 基于研究的答案: {summary_answer}")
                print(f"[问题 {i}] 基于记忆的答案: {memory_answer}")
                
                qa_result = {
                    "question": q,
                    "gold_answer": gold,
                    "category": cat,
                    "research_summary": research_summary,
                    "summary_answer": summary_answer,
                    "memory_answer": memory_answer,
                    "iterations": len(result.raw_memory.get("iterations", [])),
                    "research_trace_file": trace_file
                }
                return qa_result
            
            except Exception as e:
                print(f"[问题 {i}] [ERROR] 处理问题失败: {e}")
                import traceback
                traceback.print_exc()
                qa_result = {
                    "question": q,
                    "gold_answer": gold,
                    "category": cat,
                    "error": str(e)
                }
                return qa_result
        
        # 并行处理所有问题
        qa_items_with_index = [(i, qi) for i, qi in enumerate(qas, 1)]
        
        print(f"使用 {thread_count} 个线程并行处理 {len(qa_items_with_index)} 个问题...")
        
        qa_results = []
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            results_list = list(tqdm(
                executor.map(process_question, qa_items_with_index),
                total=len(qa_items_with_index),
                desc="处理问题"
            ))
        
        # 过滤掉None结果（category==5的问题）
        qa_results = [r for r in results_list if r is not None]
        
        # 保存结果
        results_file = os.path.join(sample_results_dir, "qa_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(qa_results, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] 结果已保存到: {results_file}")
        
        # 保存所有研究轨迹的汇总
        all_research_traces = []
        for i, qa_result in enumerate(qa_results, 1):
            if "research_trace_file" in qa_result:
                trace_file = qa_result["research_trace_file"]
                if os.path.exists(trace_file):
                    with open(trace_file, 'r', encoding='utf-8') as f:
                        trace_data = json.load(f)
                        all_research_traces.append({
                            "question_index": i,
                            "question": qa_result["question"],
                            "category": qa_result["category"],
                            "research_trace": trace_data
                        })
        
        if all_research_traces:
            traces_summary_file = os.path.join(sample_results_dir, "all_research_traces.json")
            with open(traces_summary_file, 'w', encoding='utf-8') as f:
                json.dump(all_research_traces, f, ensure_ascii=False, indent=2)
            print(f"[OK] 所有研究轨迹汇总已保存到: {traces_summary_file}")
        
        # 总结
        print(f"\n{'='*60}")
        print("处理完成统计")
        print(f"{'='*60}")
        print(f"样本ID: {sample_id}")
        print(f"会话数: {len(session_chunks)}")
        print(f"记忆摘要数: {len(final_state.abstracts)}")
        print(f"处理问题数: {len(qa_results)}")
        print(f"研究轨迹文件数: {len(all_research_traces)}")
        print(f"结果保存到: {sample_results_dir}")
        print(f"  - QA结果: qa_results.json")
        print(f"  - 记忆状态: memory_state.json")
        print(f"  - 研究轨迹汇总: all_research_traces.json")
        print(f"  - 单个研究轨迹: research_trace_q*.json")
        
        return qa_results
        
    except Exception as e:
        error_msg = f"处理样本 {sample_index} 时出错: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return []


# ========== 主函数 ==========

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="GAM 框架 + LoCoMo 数据集测试")
    parser.add_argument("--data", type=str, default="/share/project/bingyu/datasets/locomo/locomo10.json", 
                        help="LoCoMo 数据集路径")
    parser.add_argument("--outdir", type=str, default="/share/project/bingyu/code/general-agentic-memory/results/locomo_output",
                        help="输出目录")
    parser.add_argument("--start-idx", type=int, default=0, help="开始样本索引")
    parser.add_argument("--end-idx", type=int, default=None, help="结束样本索引（不包含），None表示处理所有样本")
    parser.add_argument("--thread-count", type=int, default=40, help="并行处理问题的线程数（每个样本内部）")
    parser.add_argument("--memory-model-api", type=str, default="http://localhost:8000/v1", help="记忆模型API")
    args = parser.parse_args()
    
    print("=" * 60)
    print("GAM 框架 + LoCoMo 数据集测试")
    print("=" * 60)
    print(f"数据集: {args.data}")
    print(f"输出目录: {args.outdir}")
    print(f"样本范围: {args.start_idx} 到 {args.end_idx-1 if args.end_idx else '全部'} (共 {args.end_idx - args.start_idx if args.end_idx else '全部'} 个样本)")
    print("=" * 60)
    
    # 加载数据
    samples = load_locomo(args.data)
    print(f"共加载 {len(samples)} 个样本")
    
    # 重新设置结束索引（在加载数据后）
    if args.end_idx is None:
        args.end_idx = len(samples)
    
    print(f"实际处理范围: {args.start_idx} 到 {args.end_idx-1} (共 {args.end_idx - args.start_idx} 个样本)")
    
    # 验证索引范围
    if args.start_idx < 0 or args.start_idx >= len(samples):
        print(f"错误: 开始样本索引 {args.start_idx} 超出范围 (总样本数: {len(samples)})")
        return
    
    if args.end_idx > len(samples):
        print(f"警告: 结束样本索引 {args.end_idx} 超出范围，调整为 {len(samples)}")
        args.end_idx = len(samples)
    
    if args.start_idx >= args.end_idx:
        print(f"错误: 开始索引 {args.start_idx} 必须小于结束索引 {args.end_idx}")
        return
    
    # 顺序处理每个样本，每个样本内部并行处理问题
    sample_indices = list(range(args.start_idx, args.end_idx))
    thread_count = args.thread_count
    
    print(f"将顺序处理 {len(sample_indices)} 个样本，每个样本内部使用 {thread_count} 个线程并行处理问题...")
    
    all_results = []
    
    # 顺序处理每个样本
    for sample_idx in tqdm(sample_indices, desc="处理样本"):
        sample = samples[sample_idx]
        print(f"\n{'='*80}")
        print(f"开始处理样本 {sample_idx}/{len(samples)-1} (范围: {args.start_idx}-{args.end_idx-1})")
        print(f"{'='*80}")
        
        try:
            results = process_sample(sample, sample_idx, args.outdir, args.memory_model_api, thread_count)
            print(f"[OK] 样本 {sample_idx} 处理完成")
            all_results.extend(results)
        except Exception as e:
            print(f"[ERROR] 样本 {sample_idx} 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存所有结果汇总
    if all_results:
        summary_file = os.path.join(args.outdir, f"batch_results_{args.start_idx}_{args.end_idx-1}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] 批量结果汇总已保存: {summary_file}")
    
    print(f"\n{'='*60}")
    print("[OK] 批量测试完成！")
    print(f"处理样本数: {args.end_idx - args.start_idx}")
    print(f"成功处理: {len(all_results)} 个问题")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()


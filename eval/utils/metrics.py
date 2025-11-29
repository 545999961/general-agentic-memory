# -*- coding: utf-8 -*-
"""
评估指标工具模块

提供常用的 NLP 评估指标：
- Exact Match (EM)
- F1 Score
- BLEU-1
- ROUGE
"""

import re
import string
import math
from collections import Counter
from typing import List, Dict, Any, Union


def normalize_answer(s: str) -> str:
    """
    标准化答案文本
    
    - 转小写
    - 移除标点
    - 移除冠词
    - 标准化空白字符
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_text_ruler(text: str) -> str:
    """
    RULER 数据集特定的文本标准化
    与原 old/ruler_test.py 中的 normalize_text 保持一致
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def evaluate_answer_ruler(model_response: str, ground_truth_outputs: List[str]) -> bool:
    """
    RULER 数据集特定的答案评估方法
    与原 old/ruler_test.py 中的 evaluate_answer 保持一致
    
    使用复杂的匹配逻辑：
    1. 直接包含匹配（小写）
    2. 标准化后包含匹配
    3. 词级别匹配（所有重要词都在答案中）
    """
    if not ground_truth_outputs or not model_response:
        return False

    model_response_lower = model_response.lower()
    model_response_normalized = normalize_text_ruler(model_response)

    unique_answers = list(dict.fromkeys(ground_truth_outputs))

    for answer in unique_answers:
        answer_str = str(answer).strip()
        if not answer_str:
            continue

        answer_lower = answer_str.lower()
        # 检查1: 直接包含匹配
        if answer_lower in model_response_lower:
            continue

        answer_normalized = normalize_text_ruler(answer_str)
        # 检查2: 标准化后包含匹配
        if answer_normalized in model_response_normalized:
            continue

        # 检查3: 词级别匹配（所有重要词都在答案中）
        answer_words = [w for w in answer_normalized.split() if len(w) > 2]
        if answer_words and all(word in model_response_normalized for word in answer_words):
            continue

        return False

    return True


def exact_match_score(prediction: str, ground_truths: List[str]) -> float:
    """
    计算 Exact Match 分数
    
    Args:
        prediction: 预测答案
        ground_truths: 标准答案列表
        
    Returns:
        1.0 如果匹配任一标准答案，否则 0.0
    """
    normalized_pred = normalize_answer(prediction)
    for gt in ground_truths:
        if normalized_pred == normalize_answer(gt):
            return 1.0
    return 0.0


def f1_score(prediction: str, ground_truths: List[str]) -> float:
    """
    计算 F1 分数（基于词级别）
    
    Args:
        prediction: 预测答案
        ground_truths: 标准答案列表
        
    Returns:
        最高的 F1 分数
    """
    normalized_pred = normalize_answer(prediction)
    pred_tokens = normalized_pred.split()
    
    max_f1 = 0.0
    for gt in ground_truths:
        normalized_gt = normalize_answer(gt)
        gt_tokens = normalized_gt.split()
        
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            f1 = 0.0
        else:
            precision = num_same / len(pred_tokens)
            recall = num_same / len(gt_tokens)
            f1 = 2 * precision * recall / (precision + recall)
        
        max_f1 = max(max_f1, f1)
    
    return max_f1


def rouge_score(
    prediction: str, 
    ground_truths: List[str],
    rouge_type: str = "rouge-l"
) -> float:
    """
    计算 ROUGE 分数
    
    Args:
        prediction: 预测答案
        ground_truths: 标准答案列表
        rouge_type: ROUGE 类型 ("rouge-1", "rouge-2", "rouge-l")
        
    Returns:
        最高的 ROUGE F1 分数
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
        
        max_score = 0.0
        for gt in ground_truths:
            scores = scorer.score(gt, prediction)
            score = scores[rouge_type].fmeasure
            max_score = max(max_score, score)
        
        return max_score
    except ImportError:
        # 如果 rouge_score 不可用，返回 F1 分数作为替代
        return f1_score(prediction, ground_truths)


def compute_metrics(
    predictions: List[str],
    ground_truths_list: List[List[str]],
    metrics: List[str] = ["em", "f1"]
) -> Dict[str, float]:
    """
    批量计算评估指标
    
    Args:
        predictions: 预测答案列表
        ground_truths_list: 标准答案列表的列表
        metrics: 要计算的指标列表 ["em", "f1", "rouge"]
        
    Returns:
        各指标的平均分数
    """
    if len(predictions) != len(ground_truths_list):
        raise ValueError("预测数量与标准答案数量不匹配")
    
    results = {metric: [] for metric in metrics}
    
    for pred, gts in zip(predictions, ground_truths_list):
        if not isinstance(gts, list):
            gts = [gts]
        
        if "em" in metrics:
            results["em"].append(exact_match_score(pred, gts))
        
        if "f1" in metrics:
            results["f1"].append(f1_score(pred, gts))
        
        if "rouge" in metrics or "rouge-l" in metrics:
            results.setdefault("rouge-l", []).append(rouge_score(pred, gts, "rouge-l"))
        
        if "rouge-1" in metrics:
            results.setdefault("rouge-1", []).append(rouge_score(pred, gts, "rouge-1"))
        
        if "rouge-2" in metrics:
            results.setdefault("rouge-2", []).append(rouge_score(pred, gts, "rouge-2"))
    
    # 计算平均值
    avg_results = {
        metric: sum(scores) / len(scores) if scores else 0.0
        for metric, scores in results.items()
    }
    
    return avg_results


def compute_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """
    计算准确率（严格匹配）
    
    Args:
        predictions: 预测答案列表
        ground_truths: 标准答案列表
        
    Returns:
        准确率
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("预测数量与标准答案数量不匹配")
    
    correct = sum(
        1 for pred, gt in zip(predictions, ground_truths)
        if normalize_answer(pred) == normalize_answer(gt)
    )
    
    return correct / len(predictions) if predictions else 0.0


# ========== LoCoMo 特定的评估指标 ==========

def normalize_text_locomo(s: str) -> str:
    """
    LoCoMo 数据集特定的文本标准化
    与原 eval/evalution_locomo.py 中的 normalize_text 保持一致
    """
    if s is None:
        return ""
    s = str(s)
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)  # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(^|\s)(a|an|the)(\s|$)", " ", s)  # drop english articles
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokens_locomo(s: str):
    """LoCoMo 分词"""
    s = normalize_text_locomo(s)
    return s.split() if s else []


def f1_score_locomo(pred: str, gold: str) -> float:
    """
    LoCoMo 数据集特定的 F1 计算
    与原 eval/evalution_locomo.py 中的 f1_score 保持一致
    """
    gtoks = tokens_locomo(gold)
    ptoks = tokens_locomo(pred)
    if not gtoks and not ptoks:
        return 1.0
    if not gtoks or not ptoks:
        return 0.0
    gcount = Counter(gtoks)
    pcount = Counter(ptoks)
    overlap = sum(min(pcount[t], gcount[t]) for t in pcount)
    if overlap == 0:
        return 0.0
    precision = overlap / len(ptoks)
    recall = overlap / len(gtoks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bleu1_score(pred: str, gold: str) -> float:
    """
    计算 BLEU-1 分数
    与原 eval/evalution_locomo.py 中的 bleu1_score 保持一致
    """
    gtoks = tokens_locomo(gold)
    ptoks = tokens_locomo(pred)
    if len(ptoks) == 0:
        return 0.0
    gcount = Counter(gtoks)
    pcount = Counter(ptoks)
    clipped = sum(min(pcount[t], gcount[t]) for t in pcount)
    precision = clipped / len(ptoks) if ptoks else 0.0
    if ptoks and gtoks:
        bp = 1.0 if len(ptoks) >= len(gtoks) else math.exp(1 - len(gtoks)/len(ptoks))
    else:
        bp = 0.0
    return bp * precision


def compute_locomo_metrics(
    predictions: List[str],
    ground_truths_list: List[List[str]]
) -> Dict[str, float]:
    """
    计算 LoCoMo 特定的评估指标（F1 和 BLEU-1）
    
    Args:
        predictions: 预测答案列表
        ground_truths_list: 标准答案列表的列表
        
    Returns:
        评估指标字典
    """
    if len(predictions) != len(ground_truths_list):
        raise ValueError("预测数量与标准答案数量不匹配")
    
    f1_scores = []
    bleu1_scores = []
    
    for pred, gts in zip(predictions, ground_truths_list):
        if not isinstance(gts, list):
            gts = [gts]
        
        # 对每个标准答案计算分数，取最大值
        max_f1 = max([f1_score_locomo(pred, gt) for gt in gts]) if gts else 0.0
        max_bleu1 = max([bleu1_score(pred, gt) for gt in gts]) if gts else 0.0
        
        f1_scores.append(max_f1)
        bleu1_scores.append(max_bleu1)
    
    return {
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "bleu1": sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0.0,
    }


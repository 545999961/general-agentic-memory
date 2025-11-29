# -*- coding: utf-8 -*-
"""GAM 评估工具模块"""

from eval.utils.chunking import (
    chunk_text_by_tokens,
    chunk_text_by_sentences,
    chunk_text_smartly,
)

from eval.utils.metrics import (
    normalize_answer,
    f1_score,
    exact_match_score,
    rouge_score,
    compute_metrics,
    # RULER specific
    normalize_text_ruler,
    evaluate_answer_ruler,
    # NarrativeQA specific
    bleu1_score_narrativeqa,
    # LoCoMo specific
    normalize_text_locomo,
    f1_score_locomo,
    bleu1_score,
    compute_locomo_metrics,
)

__all__ = [
    # Chunking
    "chunk_text_by_tokens",
    "chunk_text_by_sentences", 
    "chunk_text_smartly",
    # Metrics - General
    "normalize_answer",
    "f1_score",
    "exact_match_score",
    "rouge_score",
    "compute_metrics",
    # Metrics - LoCoMo specific
    "normalize_text_locomo",
    "f1_score_locomo",
    "bleu1_score",
    "compute_locomo_metrics",
]


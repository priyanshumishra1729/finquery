"""Quantitative metric helpers for FinQuery model evaluation."""

from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip().lower()


def tokenize(value: str) -> set[str]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", normalize_text(value)).strip()
    return {token for token in cleaned.split() if token}


def keyword_overlap_ratio(answer: str, reference: str) -> float:
    a_tokens = tokenize(answer)
    r_tokens = tokenize(reference)
    if not a_tokens or not r_tokens:
        return 0.0
    overlap = len(a_tokens & r_tokens)
    return overlap / max(len(r_tokens), 1)


def correctness_score(answer: str, reference: str) -> int:
    if not answer:
        return 0

    score = keyword_overlap_ratio(answer, reference)
    if score >= 0.7:
        return 3
    if score >= 0.4:
        return 2
    if score >= 0.15:
        return 1
    return 0


def relevance_score(answer: str, question: str) -> int:
    if not answer:
        return 0
    q_tokens = tokenize(question)
    a_tokens = tokenize(answer)
    if not q_tokens:
        return 2 if answer.strip() else 0
    overlap = len(q_tokens & a_tokens)
    if overlap >= max(2, len(q_tokens) // 4):
        return 2
    if overlap > 0:
        return 1
    return 0


def compute_retrieval_scores(retrieved_chunks: list[dict[str, Any]], expected_files: list[str], question: str) -> dict[str, float | str | None]:
    if not retrieved_chunks:
        return {"retrieval_precision": 0.0, "retrieval_recall": 0.0, "relevant_count": 0, "total_relevant": 0}

    q_tokens = tokenize(question)
    expected_set = {name.lower() for name in expected_files}
    relevant_retrieved = 0
    total_relevant = max(len(expected_set), 1)

    for chunk in retrieved_chunks:
        filename = str(chunk.get("filename", "")).lower()
        content = str(chunk.get("content", "")).lower()
        is_relevant = filename in expected_set or bool(q_tokens & tokenize(content))
        if is_relevant:
            relevant_retrieved += 1

    total_retrieved = len(retrieved_chunks)
    precision = relevant_retrieved / total_retrieved if total_retrieved else 0.0
    recall = relevant_retrieved / total_relevant if total_relevant else 0.0

    return {
        "retrieval_precision": precision,
        "retrieval_recall": recall,
        "relevant_count": relevant_retrieved,
        "total_relevant": total_relevant,
    }


def hallucination_indicator(answer: str, context: str, reference_answer: str) -> bool:
    if not answer:
        return False
    answer_tokens = set(tokenize(answer))
    context_tokens = set(tokenize(context))
    reference_tokens = set(tokenize(reference_answer))
    allowed_tokens = context_tokens | reference_tokens
    unsupported = answer_tokens - allowed_tokens
    if not unsupported:
        return False
    ratio = len(unsupported) / max(len(answer_tokens), 1)
    return ratio >= 0.2


def summarize_latency(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg_latency_ms": 0.0, "median_latency_ms": 0.0, "min_latency_ms": 0.0, "max_latency_ms": 0.0}
    return {
        "avg_latency_ms": statistics.mean(values),
        "median_latency_ms": statistics.median(values),
        "min_latency_ms": min(values),
        "max_latency_ms": max(values),
    }


def summarize_tokens(values: list[int]) -> dict[str[float] | None]:
    if not values:
        return {"avg_input_tokens": None, "avg_output_tokens": None, "avg_total_tokens": None}
    return {
        "avg_input_tokens": sum(values) / len(values),
        "avg_output_tokens": sum(values) / len(values),
        "avg_total_tokens": sum(values) / len(values),
    }


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_comparison_table(results_by_model: dict[str, dict[str, Any]]) -> str:
    metrics = [
        ("Accuracy", "accuracy"),
        ("Avg Correctness", "avg_correctness"),
        ("Relevance", "avg_relevance"),
        ("Retrieval Precision", "avg_retrieval_precision"),
        ("Retrieval Recall", "avg_retrieval_recall"),
        ("Hallucination Rate", "hallucination_rate"),
        ("Test Pass Rate", "test_pass_rate"),
        ("Avg Latency", "avg_latency_ms"),
        ("Median Latency", "median_latency_ms"),
        ("Avg Output Tokens", "avg_output_tokens"),
        ("Avg CPU", "avg_cpu_percent"),
        ("Peak Memory", "peak_memory_mb"),
    ]

    lines = [
        "| Metric | CodeLlama | StarCoder2 | Qwen2.5-Coder |",
        "|--------|-----------|------------|---------------|",
    ]

    for label, key in metrics:
        row: list[str] = [label]
        for model_name in ["codellama:latest", "starcoder2:3b", "qwen2.5-coder:3b"]:
            model_result = results_by_model.get(model_name, {})
            value = model_result.get(key)
            row.append(format_metric(value))
        lines.append(f"| {' | '.join(row)} |")
    return "\n".join(lines)


def format_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

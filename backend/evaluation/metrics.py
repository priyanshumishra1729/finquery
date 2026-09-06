"""Metrics and scoring helpers for FinQuery evaluation."""

from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip().lower()


def tokenize(value: str) -> set[str]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", normalize_text(value)).strip()
    return {token for token in cleaned.split() if token}


def correctness_score(answer: str, reference_answer: str) -> int:
    """0=incorrect, 1=partially correct, 2=fully correct."""
    if not answer or not answer.strip():
        return 0
    reference = normalize_text(reference_answer)
    answer_norm = normalize_text(answer)
    overlap = len(tokenize(answer_norm) & tokenize(reference))
    ref_tokens = len(tokenize(reference))
    coverage = overlap / ref_tokens if ref_tokens else 0.0
    if coverage >= 0.75 and len(answer_norm) > 20:
        return 2
    if coverage >= 0.35:
        return 1
    return 0


def relevance_score(answer: str, question: str) -> int:
    """0=irrelevant, 1=partially relevant, 2=direct answer."""
    if not answer or not answer.strip():
        return 0
    answer_tokens = tokenize(answer)
    question_tokens = tokenize(question)
    if not question_tokens:
        return 2
    overlap = len(answer_tokens & question_tokens)
    if overlap >= max(2, len(question_tokens) // 3):
        return 2
    if overlap > 0:
        return 1
    return 0


def retrieval_precision(retrieved_chunks: list[dict[str, Any]], expected_files: list[str]) -> float:
    """Return relevant_retrieved / total_retrieved."""
    if not retrieved_chunks:
        return 0.0
    expected = {str(item).lower() for item in expected_files}
    relevant = 0
    for chunk in retrieved_chunks:
        filename = str(chunk.get("filename", "")).lower()
        content = str(chunk.get("content", "")).lower()
        if filename in expected or any(token in content for token in ["balance sheet", "cash flow", "ratio", "equation", "financial", "statement"]):
            relevant += 1
    return relevant / len(retrieved_chunks)


def retrieval_recall(retrieved_chunks: list[dict[str, Any]], expected_files: list[str]) -> float:
    """Return relevant_retrieved / expected_relevant_chunks. When expected context is unavailable, return 0.0 rather than inventing data."""
    if not expected_files:
        return 0.0
    expected = {str(item).lower() for item in expected_files}
    relevant_retrieved = 0
    for chunk in retrieved_chunks:
        filename = str(chunk.get("filename", "")).lower()
        if filename in expected:
            relevant_retrieved += 1
    return relevant_retrieved / len(expected)


def hallucination_indicator(answer: str, reference_answer: str, context: str) -> bool:
    """Heuristic: if a substantial part of the answer is not present in either the context or expected reference answer, flag it as hallucinated."""
    if not answer or not answer.strip():
        return False
    answer_tokens = set(tokenize(answer))
    reference_tokens = set(tokenize(reference_answer))
    context_tokens = set(tokenize(context))
    allowed = answer_tokens & (reference_tokens | context_tokens)
    unsupported = answer_tokens - (reference_tokens | context_tokens)
    if not answer_tokens:
        return False
    return len(unsupported) / len(answer_tokens) >= 0.3 and len(allowed) < len(answer_tokens)


def summarize_series(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"avg": None, "median": None, "min": None, "max": None}
    return {
        "avg": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "N/A") for key in fieldnames})

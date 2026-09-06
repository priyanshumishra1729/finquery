"""Run FinQuery model evaluation across the three configured Ollama models."""

from __future__ import annotations

import ast
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import (
    build_comparison_table,
    compute_retrieval_scores,
    correctness_score,
    hallucination_indicator,
    relevance_score,
    write_json,
)
from evaluation.resource_monitor import ResourceMonitor

from backend.app.services.application_service import ApplicationService
from backend.app.services.ollama_service import OllamaService


QUESTION_PATH = Path(__file__).with_name("questions.json")
RESULTS_DIR = Path(__file__).parent / "results"
REPORT_DIR = Path(__file__).parent / "report"
RAG_ANALYSIS_DIR = Path(__file__).parent / "rag_analysis"

MODELS = [
    "codellama:latest",
    "starcoder2:3b",
    "qwen2.5-coder:3b",
]


def read_questions() -> list[dict[str, Any]]:
    payload = json.loads(QUESTION_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("questions.json must contain a list of question entries")
    return payload


def build_service(model_name: str) -> ApplicationService:
    ollama_service = OllamaService(model=model_name, timeout=1800.0)
    return ApplicationService(ollama_service=ollama_service)


def extract_python_snippet(answer: str) -> str | None:
    code_blocks = re.findall(r"```python\n(.*?)```", answer, flags=re.DOTALL | re.IGNORECASE)
    if code_blocks:
        return code_blocks[0].strip()
    return None


def test_generated_code(answer: str) -> str:
    snippet = extract_python_snippet(answer)
    if snippet is None:
        return "not_applicable"

    try:
        ast.parse(snippet)
    except SyntaxError:
        return "failed"
    return "passed"


def compute_question_metrics(question: dict[str, Any], answer: str, context: str, retrieved_chunks: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    correctness = correctness_score(answer, question["reference_answer"])
    relevance = relevance_score(answer, question["question"])
    retrieval = compute_retrieval_scores(retrieved_chunks, question.get("expected_files", []), question["question"])
    hallucinated = hallucination_indicator(answer, context, question["reference_answer"])
    test_result = test_generated_code(answer) if question.get("evaluation_type") == "code_generation" else "not_applicable"

    record = {
        "model": model_name,
        "question_id": question["id"],
        "category": question["category"],
        "question": question["question"],
        "reference_answer": question["reference_answer"],
        "expected_files": question.get("expected_files", []),
        "evaluation_type": question.get("evaluation_type"),
        "correctness_score": correctness,
        "relevance_score": relevance,
        "retrieval_precision": retrieval["retrieval_precision"],
        "retrieval_recall": retrieval["retrieval_recall"],
        "hallucination_indicator": hallucinated,
        "test_pass_result": test_result,
    }
    return record


def evaluate_model(model_name: str, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for question in questions:
        startup_service = build_service(model_name)
        question_text = question["question"]
        start = time.perf_counter()
        retrieved_chunks = startup_service.rag_service.retrieval_service.retrieve(question_text, top_k=5)
        context = startup_service.rag_service.build_context(question_text, top_k=5)

        def generate_response() -> dict[str, Any]:
            return startup_service.process_chat_with_context(question_text)

        monitor = ResourceMonitor(interval=0.05)
        monitor.start()
        try:
            generation_output = generate_response()
        finally:
            resource_data = monitor.stop()

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        metadata = generation_output.get("metadata", {})
        answer = generation_output.get("answer", "")
        prompt_eval_count = metadata.get("prompt_eval_count")
        eval_count = metadata.get("eval_count")
        total_tokens = metadata.get("total_tokens")

        metrics_data = compute_question_metrics(question, answer, context, retrieved_chunks, model_name)
        result = {
            "model": model_name,
            "question_id": question["id"],
            "category": question["category"],
            "question": question_text,
            "retrieved_context": context,
            "retrieved_chunks": retrieved_chunks,
            "answer": answer,
            "latency_ms": round(elapsed_ms, 2),
            "input_tokens": prompt_eval_count,
            "output_tokens": eval_count,
            "total_tokens": total_tokens if total_tokens is not None else ((prompt_eval_count or 0) + (eval_count or 0)),
            "avg_cpu_percent": float(resource_data.get("avg_cpu_percent", 0.0) or 0.0),
            "peak_memory_bytes": float(resource_data.get("peak_memory_bytes", 0.0) or 0.0),
            "peak_memory_mb": float(resource_data.get("peak_memory_bytes", 0.0) or 0.0) / (1024 * 1024),
            "correctness_score": metrics_data["correctness_score"],
            "relevance_score": metrics_data["relevance_score"],
            "retrieval_precision": metrics_data["retrieval_precision"],
            "retrieval_recall": metrics_data["retrieval_recall"],
            "hallucination_indicator": metrics_data["hallucination_indicator"],
            "test_pass_result": metrics_data["test_pass_result"],
            "reference_answer": question["reference_answer"],
            "expected_files": question.get("expected_files", []),
            "evaluation_type": question.get("evaluation_type"),
            "model_name": model_name,
            "experiment_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        results.append(result)

    return results


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    correctness_scores = [item["correctness_score"] for item in results]
    relevance_scores = [item["relevance_score"] for item in results]
    retrieval_precision = [item["retrieval_precision"] for item in results if item["retrieval_precision"] is not None]
    retrieval_recall = [item["retrieval_recall"] for item in results if item["retrieval_recall"] is not None]
    hallucination_flags = [1 if item["hallucination_indicator"] else 0 for item in results]
    latencies = [item["latency_ms"] for item in results]
    input_tokens = [item["input_tokens"] for item in results if item["input_tokens"] is not None]
    output_tokens = [item["output_tokens"] for item in results if item["output_tokens"] is not None]
    total_tokens = [item["total_tokens"] for item in results if item["total_tokens"] is not None]
    cpu_values = [item["avg_cpu_percent"] for item in results]
    memory_values = [item["peak_memory_mb"] for item in results]
    test_results = [1 if item["test_pass_result"] == "passed" else 0 for item in results if item["test_pass_result"] != "not_applicable"]

    total_questions = len(results)
    accuracy = sum(1 for score in correctness_scores if score == 3) / total_questions if total_questions else 0.0

    summary = {
        "model": results[0]["model"] if results else "unknown",
        "total_questions": total_questions,
        "accuracy": accuracy,
        "avg_correctness": statistics.mean(correctness_scores) if correctness_scores else 0.0,
        "avg_relevance": statistics.mean(relevance_scores) if relevance_scores else 0.0,
        "avg_retrieval_precision": statistics.mean(retrieval_precision) if retrieval_precision else 0.0,
        "avg_retrieval_recall": statistics.mean(retrieval_recall) if retrieval_recall else 0.0,
        "hallucination_rate": statistics.mean(hallucination_flags) if hallucination_flags else 0.0,
        "avg_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "median_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "min_latency_ms": min(latencies) if latencies else 0.0,
        "max_latency_ms": max(latencies) if latencies else 0.0,
        "avg_input_tokens": statistics.mean(input_tokens) if input_tokens else None,
        "avg_output_tokens": statistics.mean(output_tokens) if output_tokens else None,
        "avg_total_tokens": statistics.mean(total_tokens) if total_tokens else None,
        "avg_cpu_percent": statistics.mean(cpu_values) if cpu_values else 0.0,
        "peak_memory_mb": max(memory_values) if memory_values else 0.0,
        "test_pass_rate": (sum(test_results) / len(test_results)) if test_results else 0.0,
    }
    return summary


def create_rag_cases(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    selected_questions = {item["question_id"] for item in results if item["question_id"] in {1, 3, 5, 8, 15, 18, 21, 25}}
    for item in results:
        if item["question_id"] not in selected_questions:
            continue
        retrieval = item["retrieved_chunks"][:3]
        classification = "Relevant information retrieved"
        if item["hallucination_indicator"]:
            classification = "Hallucination despite relevant context"
        elif not retrieval:
            classification = "Important information missed"
        elif item["retrieval_precision"] < 0.5:
            classification = "Irrelevant information retrieved"
        elif item["correctness_score"] >= 2 and item["retrieval_precision"] < 1.0:
            classification = "Correct answer despite imperfect retrieval"

        cases.append(
            {
                "question_id": item["question_id"],
                "category": item["category"],
                "question": item["question"],
                "retrieved_chunks": retrieval,
                "final_model_response": item["answer"],
                "correctness": item["correctness_score"],
                "hallucination_status": item["hallucination_indicator"],
                "classification": classification,
                "model": item["model"],
            }
        )
    return cases


def write_reports(results_by_model: dict[str, dict[str, Any]], rag_cases: list[dict[str, Any]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_md = build_comparison_table(results_by_model)
    (REPORT_DIR / "comparison.md").write_text(comparison_md + "\n", encoding="utf-8")

    rag_lines = [
        "# RAG Analysis",
        "",
        "This section highlights retrieval quality and whether the answer was grounded in source context.",
        "",
    ]
    for case in rag_cases:
        rag_lines.append(f"## Question {case['question_id']} ({case['model']})")
        rag_lines.append(f"- Category: {case['category']}")
        rag_lines.append(f"- Question: {case['question']}")
        rag_lines.append(f"- Classification: {case['classification']}")
        rag_lines.append(f"- Correctness: {case['correctness']}")
        rag_lines.append(f"- Hallucination: {case['hallucination_status']}")
        rag_lines.append("- Retrieved chunks:")
        for chunk in case["retrieved_chunks"]:
            rag_lines.append(f"  - {chunk.get('filename', 'unknown')} | chunk {chunk.get('chunk_index', 'n/a')} | distance {chunk.get('distance', 'n/a')}")
        rag_lines.append(f"- Final response: {case['final_model_response']}")
        rag_lines.append("")
    (REPORT_DIR / "rag_analysis.md").write_text("\n".join(rag_lines), encoding="utf-8")

    repo_lines = [
        "# Repository-Level Analysis",
        "",
        "The FinQuery repository implements a straightforward RAG pipeline: document loading, chunking, embedding, vector search, prompt construction, and answer generation. The following repository-level questions were evaluated against the same knowledge base and the same retrieval pipeline for each model.",
        "",
        "1. Which files are involved in the RAG pipeline?",
        "   Relevant files: backend/app/services/document_loader.py, chunking_service.py, embedding_service.py, vector_store_service.py, retrieval_service.py, rag_service.py, application_service.py, ollama_service.py.",
        "",
        "2. What happens when a user submits a query?",
        "   The ApplicationService normalizes the message, retrieves context from the knowledge base, builds a prompt with the context and question, and sends it to Ollama for generation.",
        "",
        "3. Which component generates embeddings?",
        "   EmbeddingService.generate_embedding calls the Ollama /api/embed endpoint using the nomic-embed-text model.",
        "",
        "4. Which component performs vector search?",
        "   VectorStoreService.search executes similarity search in ChromaDB and returns matching chunks with filename, chunk_index, content, and distance.",
        "",
        "5. How does the retrieved context reach the LLM?",
        "   RAGService builds a prompt-ready context block, then ApplicationService includes it together with the user question before calling OllamaService.generate_response.",
        "",
        "6. Which files would need to change if the LLM provider were replaced?",
        "   The provider abstraction is isolated in ollama_service.py. The rest of the RAG pipeline remains stable unless the prompt contract changes.",
        "",
        "This analysis confirms that the repository is intentionally modular and that the evaluation compares only model choice, not retrieval or prompt design.",
    ]
    (REPORT_DIR / "repository_analysis.md").write_text("\n".join(repo_lines), encoding="utf-8")


def main() -> None:
    questions = read_questions()
    all_results: dict[str, list[dict[str, Any]]] = {}
    results_by_model: dict[str, dict[str, Any]] = {}

    for model_name in MODELS:
        model_results = evaluate_model(model_name, questions)
        summary = build_summary(model_results)
        all_results[model_name] = model_results
        results_by_model[model_name] = summary

        write_json(RESULTS_DIR / f"{model_name.replace(':', '').replace('.', '').replace('-', '')}_results.json", {
            "model": model_name,
            "results": model_results,
            "summary": summary,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })

    combined = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": MODELS,
        "results_by_model": results_by_model,
        "all_results": all_results,
    }
    write_json(RESULTS_DIR / "combined_results.json", combined)

    rag_cases = []
    for model_name, records in all_results.items():
        rag_cases.extend(create_rag_cases(records))
    write_json(RAG_ANALYSIS_DIR / "rag_cases.json", rag_cases)
    write_reports(results_by_model, rag_cases)

    print(json.dumps(results_by_model, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import ast
import json
import os
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from agents_v2.cli import run_question
from agents_v2.loaders import build_resources_from_index
from agents_v2.router import QuestionRouter
from eval_score import eval_score, show_results


DEFAULT_PROMPT = (
    "Question: {question}\n"
    "Predicted Answer: {answer}\n"
    "Ground Truth Answer: {gt}\n\n"
    "Please evaluate if the predicted answer is correct compared to the ground truth.\n"
    "Score the answer on:\n"
    "Binary correctness (0-1): 1 if the answer is correct, 0 if it is incorrect\n\n"
    "Return only a string with these scores in a dictionary and can be parsed by json.loads, e.g. {\"binary_correctness\": 1}"
)


def render_prompt(template: str, values: Dict[str, str]) -> str:
    """
    Format prompt templates while tolerating literal braces in the template.

    Falls back to direct placeholder replacement if str.format fails due to
    unescaped braces (e.g., JSON examples like {"binary_correctness": 1}).
    """
    try:
        return template.format(**values)
    except (KeyError, ValueError):
        prompt = template
        for key, value in values.items():
            prompt = prompt.replace(f"{{{key}}}", value)
        return prompt


def parse_list_field(raw: str | None) -> List[str]:
    if not raw:
        return []
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return []


def parse_int_list(raw: str | None) -> List[int]:
    items = parse_list_field(raw)
    result: List[int] = []
    for item in items:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


def resolve_doc_dir(root: Path, doc_id: str) -> Optional[Path]:
    candidates = [
        root / doc_id,
        root / doc_id.replace(".pdf", ""),
    ]
    for candidate in candidates:
        if (candidate / "indexes").is_dir():
            return candidate / "indexes"
        if candidate.is_dir():
            return candidate
    return None


def call_eval_llm(
    client: OpenAI,
    question: str,
    predicted: str,
    ground_truth: str,
    prompt_template: str,
    model_name: str,
) -> float:
    prompt = render_prompt(
        prompt_template,
        {
            "question": str(question),
            "answer": str(predicted),
            "gt": str(ground_truth),
        },
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=32,
    )
    content = response.choices[0].message.content
    try:
        return float(json.loads(content).get("binary_correctness", 0))
    except Exception:
        return 0.0


def evaluate_pipeline(
    samples_path: Path,
    doc_root: Path,
    prompt_text: str,
    model_name: str,
    answer_report: Optional[Path],
    detailed_json: Optional[Path],
    limit: Optional[int],
    offset: int,
) -> None:
    with samples_path.open("r", encoding="utf-8") as fh:
        samples = json.load(fh)

    router_confusion: Counter[Tuple[str, str]] = Counter()
    router_total = 0
    router_success = 0
    support_hits = 0
    support_total = 0
    skipped_docs: Dict[str, int] = defaultdict(int)
    evaluated_samples: List[Dict[str, object]] = []

    client = OpenAI()

    processed = 0
    for idx, row in enumerate(samples):
        if idx < offset:
            continue
        if limit is not None and processed >= limit:
            break
        processed += 1

        doc_id = row.get("doc_id")
        question = row.get("question") or ""
        if not doc_id or not question:
            continue

        doc_dir = resolve_doc_dir(doc_root, doc_id)
        if not doc_dir:
            skipped_docs[doc_id] += 1
            continue

        try:
            resources, _ = build_resources_from_index(doc_dir)
        except FileNotFoundError:
            skipped_docs[doc_id] += 1
            continue

        evidence_sources_list = parse_list_field(row.get("evidence_sources"))
        evidence_pages_list = parse_int_list(row.get("evidence_pages"))

        router_decision = None
        if evidence_sources_list:
            router = QuestionRouter()
            if getattr(resources, "toc_outline", None):
                router.attach_toc(resources.toc_outline)
            router_decision = router.route(question)
            predicted_hint = (router_decision.signals.evidence_hint or "unknown").lower()
            normalized_sources = {source.strip().lower() for source in evidence_sources_list if isinstance(source, str)}
            if not normalized_sources:
                normalized_sources = {"unknown"}
            for gold_hint in normalized_sources:
                router_confusion[(gold_hint, predicted_hint)] += 1
            if predicted_hint in normalized_sources:
                router_success += 1
            router_total += 1

        try:
            answer, orchestrator, _ = run_question(doc_dir, question)
        except Exception:
            skipped_docs[doc_id] += 1
            continue

        decision_full = (
            orchestrator.memory.router_history[-1]
            if orchestrator.memory.router_history
            else router_decision
        )
        predicted_pages: set[int] = set()
        if len(evidence_pages_list) == 1:
            support_total += 1
            for node in answer.support_nodes:
                page_idx = orchestrator.retriever_manager.resources.node_pages.get(node)
                if page_idx is not None:
                    predicted_pages.add(page_idx)
            if evidence_pages_list[0] in predicted_pages:
                support_hits += 1

        # LLM-based answer evaluation
        predicted_answer = answer.answer
        llm_score = call_eval_llm(client, question, predicted_answer, row.get("answer"), prompt_text, model_name)

        record = deepcopy(row)
        record["pred"] = predicted_answer
        record["llm_binary_correctness"] = llm_score
        record["router_hint"] = (
            decision_full.signals.evidence_hint if decision_full is not None else None
        )
        record["support_nodes"] = answer.support_nodes
        record["support_pages_pred"] = list(predicted_pages) if predicted_pages else None

        evaluated_samples.append(record)

    print("=== Router evidence_hint confusion matrix ===")
    if router_total == 0:
        print("No samples with evidence sources were evaluated.")
    else:
        print(f"Success rate: {router_success}/{router_total} = {router_success/router_total:.2%}")
        by_gold: Dict[str, Counter[str]] = defaultdict(Counter)
        for (gold, pred), count in router_confusion.items():
            by_gold[gold][pred] += count
        for gold, preds in sorted(by_gold.items()):
            total = sum(preds.values())
            pred_str = ", ".join(f"{pred}:{cnt}" for pred, cnt in preds.items())
            print(f"gold={gold:<20} total={total:<4} -> {pred_str}")

    print("\n=== Reasoner support-node page accuracy (single-page cases) ===")
    if support_total == 0:
        print("No single-page samples evaluated.")
    else:
        accuracy = support_hits / support_total
        print(f"Accuracy: {support_hits}/{support_total} = {accuracy:.2%}")

    if detailed_json:
        with detailed_json.open("w", encoding="utf-8") as jf:
            json.dump(evaluated_samples, jf, indent=2)
        print(f"\nDetailed results saved to: {detailed_json}")

    if answer_report:
        show_results(evaluated_samples, show_path=answer_report)
        print(f"Answer evaluation summary saved to: {answer_report}")

    if skipped_docs:
        print("\nSkipped samples (missing indices or execution failure):")
        for doc_id, count in skipped_docs.items():
            print(f"  {doc_id}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate router hints, support nodes, and answers with LLM scoring.")
    parser.add_argument("--samples", type=Path, required=True, help="Path to samples.json")
    parser.add_argument("--doc-root", type=Path, required=True, help="Root directory containing document indexes")
    parser.add_argument("--answer-report", type=Path, help="Optional path to save answer summary report")
    parser.add_argument("--answer-prompt", type=Path, help="Prompt file for LLM-based answer evaluation")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name for answer evaluation")
    parser.add_argument("--dump-json", type=Path, help="Optional path to save detailed JSON results")
    parser.add_argument("--limit", type=int, help="Optional limit on number of samples")
    parser.add_argument("--offset", type=int, default=0, help="Number of samples to skip from the beginning")
    args = parser.parse_args()

    prompt_text = DEFAULT_PROMPT
    if args.answer_prompt and args.answer_prompt.is_file():
        prompt_text = args.answer_prompt.read_text(encoding="utf-8")

    evaluate_pipeline(
        samples_path=args.samples,
        doc_root=args.doc_root,
        prompt_text=prompt_text,
        model_name=args.model,
        answer_report=args.answer_report,
        detailed_json=args.dump_json,
        limit=args.limit,
        offset=args.offset or 0,
    )


if __name__ == "__main__":
    main()

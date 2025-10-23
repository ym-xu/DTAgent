from __future__ import annotations

import re
from collections import defaultdict
from math import isclose
from typing import Dict, List, Sequence


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(
                    1 + min(distances[i1], distances[i1 + 1], distances_[-1])
                )
        distances = distances_
    return distances[-1]


def anls_compute(groundtruth: str, prediction: str, threshold: float = 0.5) -> float:
    dist = levenshtein_distance(groundtruth, prediction)
    length = max(len(groundtruth.upper()), len(prediction.upper()))
    value = 0.0 if length == 0 else float(dist) / float(length)
    anls = 1.0 - value
    if anls <= threshold:
        anls = 0.0
    return anls


def is_float_equal(
    reference: float,
    prediction: str,
    include_percentage: bool = False,
    is_close: bool = False,
) -> bool:
    def get_precision(gt_ans: float) -> int:
        precision = 3
        text = str(gt_ans)
        if "." in text:
            precision = len(text.split(".")[-1])
        return precision

    reference = float(str(reference).strip().rstrip("%").strip())
    try:
        prediction_val = float(str(prediction).strip().rstrip("%").strip())
    except Exception:
        return False

    if include_percentage:
        gt_result = [reference / 100, reference, reference * 100]
    else:
        gt_result = [reference]
    for item in gt_result:
        try:
            if is_close:
                if isclose(item, prediction_val, rel_tol=0.01):
                    return True
            precision = max(min(get_precision(prediction_val), get_precision(item)), 2)
            if round(prediction_val, precision) == round(item, precision):
                return True
        except Exception:
            continue
    return False


def get_clean_string(s: str) -> str:
    s = str(s).lower().strip()
    for suffix in ("mile", "miles", "million"):
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
    s = re.sub(r"\s*\([^)]*\)", "", s).strip()  # remove parentheses
    s = re.sub(r"^['\"]|['\"]$", "", s).strip()
    s = s.strip().lstrip("$").strip()
    s = s.strip().rstrip("%").strip()
    return s


def is_exact_match(s: str) -> bool:
    if "https://" in s:
        return True
    if s.endswith(".py") or s.endswith("ipynb"):
        return True
    if s.startswith("page"):
        return True
    if re.fullmatch(r"\b\d+(-\d+|\s\d+)?\b", s):
        return True
    if "a.m." in s or "p.m." in s:
        return True
    if re.fullmatch(r"\b\d{4}[-\s]\d{2}[-\s]\d{2}\b", s):
        return True
    if re.fullmatch(r"\b\d{4}[-\s]\d{2}\b", s):
        return True
    if re.fullmatch(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", s):
        return True
    return False


def isfloat(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def eval_score(gt, pred, answer_type: str) -> float:
    if answer_type == "Int":
        try:
            gt_int = int(gt)
            pred_int = int(float(pred))
        except Exception:
            return 0.0
        return float(gt_int == pred_int)

    if answer_type == "Float":
        try:
            gt_float = float(get_clean_string(str(gt)))
            pred_str = get_clean_string(str(pred))
        except Exception:
            return 0.0
        return float(is_float_equal(gt_float, pred_str, include_percentage=True, is_close=True))

    if answer_type in ["Str", "None"]:
        gt_clean = get_clean_string(gt)
        pred_clean = get_clean_string(pred)
        if is_exact_match(gt_clean):
            return float(gt_clean == pred_clean)
        return anls_compute(gt_clean, pred_clean)

    # List-like
    if isinstance(gt, str) and gt.startswith("["):
        gt_list = eval(gt)
    else:
        gt_list = gt if isinstance(gt, list) else [gt]

    if isinstance(pred, str) and pred.startswith("["):
        pred_list = eval(pred)
    else:
        pred_list = pred if isinstance(pred, list) else [pred]

    if len(gt_list) != len(pred_list):
        return 0.0

    gt_clean_list = sorted(get_clean_string(item) for item in gt_list)
    pred_clean_list = sorted(get_clean_string(item) for item in pred_list)

    if gt_clean_list and (isfloat(gt_clean_list[0]) or is_exact_match(gt_clean_list[0])):
        return float("-".join(gt_clean_list) == "-".join(pred_clean_list))

    scores = [
        anls_compute(gt_v, pred_v)
        for gt_v, pred_v in zip(gt_clean_list, pred_clean_list)
    ]
    return float(min(scores))


def eval_acc_and_f1(samples: Sequence[Dict[str, object]]) -> Tuple[float, float]:
    evaluated_samples = [sample for sample in samples if "score" in sample]
    if not evaluated_samples:
        return 0.0, 0.0

    acc = sum(sample["score"] for sample in evaluated_samples) / len(evaluated_samples)
    try:
        positive_samples = [
            sample for sample in evaluated_samples if sample.get("answer") != "Not answerable"
        ]
        recall = sum(sample["score"] for sample in positive_samples) / len(positive_samples)
        predicted_positive = [
            sample for sample in evaluated_samples if sample.get("pred") != "Not answerable"
        ]
        precision = sum(sample["score"] for sample in positive_samples) / len(predicted_positive)
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
    except Exception:
        f1 = 0.0

    return acc, f1


def show_results(samples: Sequence[Dict[str, object]], show_path: Path) -> None:
    def _ensure_list_field(sample: Dict[str, object], key: str) -> List[int | str]:
        value = sample.get(key)
        if isinstance(value, str) and value.startswith("["):
            return eval(value)
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return []

    processed_samples = []
    for sample in samples:
        copy_sample = dict(sample)
        copy_sample["evidence_pages"] = _ensure_list_field(sample, "evidence_pages")
        copy_sample["evidence_sources"] = _ensure_list_field(sample, "evidence_sources")
        processed_samples.append(copy_sample)

    acc, f1 = eval_acc_and_f1(processed_samples)
    with show_path.open("w", encoding="utf-8") as fh:
        fh.write(f"Overall Acc: {acc} | Question Number: {len(processed_samples)}\n")
        fh.write(f"Overall F1-score: {f1} | Question Number: {len(processed_samples)}\n")
        fh.write("-----------------------\n")

        single_page = [s for s in processed_samples if len(s["evidence_pages"]) == 1]
        cross_page = [
            s for s in processed_samples if len(s["evidence_pages"]) != 1 and s.get("answer") != "Not answerable"
        ]
        unanswerable = [s for s in processed_samples if s.get("answer") == "Not answerable"]

        fh.write(f"Single-page | Accuracy: {eval_acc_and_f1(single_page)[0]} | Question Number: {len(single_page)}\n")
        fh.write(f"Cross-page | Accuracy: {eval_acc_and_f1(cross_page)[0]} | Question Number: {len(cross_page)}\n")
        fh.write(f"Unanswerable | Accuracy: {eval_acc_and_f1(unanswerable)[0]} | Question Number: {len(unanswerable)}\n")
        fh.write("-----------------------\n")

        source_sample_dict: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        document_type_dict: Dict[str, List[Dict[str, object]]] = defaultdict(list)

        for sample in processed_samples:
            for answer_source in sample["evidence_sources"]:
                source_sample_dict[str(answer_source)].append(sample)
            if sample.get("doc_type"):
                document_type_dict[sample["doc_type"]].append(sample)

        for source, sub_samples in sorted(source_sample_dict.items()):
            fh.write(
                f"Evidence Sources: {source} | Accuracy: {eval_acc_and_f1(sub_samples)[0]} | "
                f"Question Number: {len(sub_samples)}\n"
            )

        fh.write("-----------------------\n")
        for doc_type, sub_samples in sorted(document_type_dict.items()):
            fh.write(
                f"Document Type: {doc_type} | Accuracy: {eval_acc_and_f1(sub_samples)[0]} | "
                f"Question Number: {len(sub_samples)}\n"
            )


__all__ = ["eval_score", "show_results", "eval_acc_and_f1"]

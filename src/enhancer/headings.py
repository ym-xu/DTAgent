from typing import Any, Dict, List, Optional, Tuple
import json
import re

from .toc import _reindex_flat
from ..utils.llm_clients import gpt_llm_call


_TERMINAL_PUNCT = re.compile(r"[\.!?;:。！？；：]$")


def _is_heading_text(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return False
    # basic: short-ish, no terminal punctuation
    words = s.split()
    if len(words) <= 1:
        return True
    if len(words) <= 5 and not _TERMINAL_PUNCT.search(s):
        return True
    # strong casing signal
    letters = [ch for ch in s if ch.isalpha()]
    if letters:
        up = sum(1 for ch in letters if ch.isupper())
        if up / max(1, len(letters)) >= 0.7:
            return True
    # numbered heading like "1.", "1.2", "3.2", "A.", "I."
    if re.match(r"^(\d+(?:\.\d+)*)\s+", s):
        return True
    if re.match(r"^[IVXLCM]+\.?\s+", s):
        return True
    if re.match(r"^[A-Z]\.\s+", s):
        return True
    return False


def _should_consider_merge(t1: str, t2: str) -> bool:
    a = (t1 or "").strip()
    b = (t2 or "").strip()
    if not a or not b:
        return False
    # If both are heading-like, and a looks incomplete and b looks a valid continuation
    if not _is_heading_text(a) or not _is_heading_text(b):
        return False
    # a likely incomplete fragment
    a_words = a.split()
    if len(a_words) <= 5 and not _TERMINAL_PUNCT.search(a):
        return True
    # a ends with connecting word
    tail = a_words[-1].lower() if a_words else ""
    if tail in {"of", "for", "with", "and", "the", "in", "on", "at", "to", "&"}:
        return True
    # explicit patterns
    prefix = " ".join(a_words[:2]).lower()
    if a.lower().startswith(("department of", "school of", "faculty of", "college of", "chapter", "part", "section")):
        return True
    return False


def _merge_text(a: str, b: str) -> str:
    a = (a or "").strip()
    b = (b or "").strip()
    if not a:
        return b
    if not b:
        return a
    # simple space join; collapse double spaces
    s = f"{a} {b}"
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _llm_decide_merge(t1: str, t2: str, *, model: str = "gpt-4o") -> bool:
    messages = [
        {"role": "system", "content": "You only output JSON."},
        {
            "role": "user",
            "content": (
                "You are given two adjacent heading lines extracted from a PDF.\n"
                "Decide whether they belong to the SAME single heading (i.e., the first is an incomplete fragment that continues into the second),\n"
                "or they are two separate headings.\n"
                "Rules:\n- Merge=true for patterns like 'DEPARTMENT OF' + 'Chinese Studies', 'EMBARK ON' + 'YOUR JOURNEY WITH ...'.\n"
                "- Merge=false if they are distinct headings or should not be combined.\n"
                "Output only one JSON object: {\"merge\": true|false}.\n\n"
                f"line1: {t1}\nline2: {t2}"
            ),
        },
    ]
    try:
        raw = gpt_llm_call(messages, images=None, model=model, json_mode=True)
        obj = json.loads(raw)
        return bool(obj.get("merge", False))
    except Exception:
        return False


def merge_adjacent_headings(
    flat_root: Dict[str, Any],
    *,
    use_llm: bool = True,
    llm_model: str = "gpt-4o",
) -> Dict[str, Any]:
    """
    Merge pairs of adjacent heading nodes (node_level==1, type==text) when they likely form one heading.
    Heuristic filters propose candidates; if use_llm=True, each candidate is confirmed by LLM (gpt-4o-mini).
    Returns a new flat_root with reindexed nodes and preserved meta.
    """
    if not isinstance(flat_root, dict):
        return flat_root
    children: List[Dict[str, Any]] = list(flat_root.get("children", []))
    if not children:
        return flat_root

    merged_children: List[Dict[str, Any]] = []
    i = 0
    while i < len(children):
        cur = children[i]
        nxt = children[i + 1] if i + 1 < len(children) else None
        # Default: append cur and advance by 1
        do_merge = False
        if (
            isinstance(cur, dict)
            and isinstance(nxt, dict)
            and cur.get("type") == "text"
            and nxt.get("type") == "text"
            and cur.get("node_level") == 1
            and nxt.get("node_level") == 1
            and cur.get("page_idx") == nxt.get("page_idx")
        ):
            t1 = str(cur.get("text", "") or "").strip()
            t2 = str(nxt.get("text", "") or "").strip()
            if _should_consider_merge(t1, t2):
                do_merge = True
                if use_llm:
                    do_merge = _llm_decide_merge(t1, t2, model=llm_model)

        if do_merge:
            new_text = _merge_text(cur.get("text", ""), nxt.get("text", ""))
            merged_children.append(
                {
                    "type": "text",
                    "text": new_text,
                    "page_idx": cur.get("page_idx"),
                    "node_level": 1,
                }
            )
            i += 2
        else:
            merged_children.append(cur)
            i += 1

    # Reindex and preserve meta
    doc_id = flat_root.get("doc_id") or "document"
    new_root = _reindex_flat(doc_id, merged_children, flat_root.get("source_path"), preserve_meta=flat_root)
    return new_root

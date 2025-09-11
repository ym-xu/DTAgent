from typing import Any, Dict
import json


def _compact_payload(payload: Dict[str, Any]) -> str:
    # Produce a compact JSON string of the payload to embed in prompt
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def render_plan_prompt(payload: Dict[str, Any], *, level_max: int = 4) -> str:
    """
    Render a strict prompt instructing the model to output a plan JSON object with keys:
    - page_idx: int
    - items: [{node_id: str, level: int}] (must be a full permutation of the given elements; level 0..level_max; 0 means non-title)
    - merges: [{into: node_id, from: [node_id,...]}] (optional; only adjacent text nodes)
    - virtual_titles: [{text: str, level: int, insert_after?: node_id}] (optional)
    The output must be a single JSON object only. No extra commentary, no Markdown, no code fences.
    """
    compact = _compact_payload(payload)
    return (
        "You are given a page payload with elements (node_id, type, snippet, outline).\n"
        "Task: Decide a reading order and heading levels for ALL elements on this page, optionally merging adjacent text-only headings and adding a minimal missing heading if necessary.\n"
        "Output a single JSON object with keys: page_idx, items, merges (optional), virtual_titles (optional).\n"
        f"Rules: items MUST be a full permutation of the provided node_id list. level is an integer 0..{level_max}; 0 means body (non-heading).\n"
        "Merges are allowed only for adjacent text nodes after reordering. Virtual titles must include {text, level} and optional insert_after (a node_id from this page).\n"
        "Do NOT include any commentary, Markdown, or extra keys.\n\n"
        "Payload follows on next line as JSON. Use it strictly.\n"
        + compact
    )


def render_jsonlist_prompt(payload: Dict[str, Any], *, level_max: int = 4) -> str:
    """
    Render a strict prompt instructing the model to output a jsonlist (array) of {node_id, level} covering ALL elements.
    The output must be a single JSON array only. No extra commentary, no Markdown, no code fences.
    """
    compact = _compact_payload(payload)
    return (
        "You are given a page payload with elements (node_id, type, snippet, outline).\n"
        "Task: Decide a reading order and heading levels for ALL elements on this page.\n"
        f"Output a single JSON array only: [{{node_id: string, level: int}}]. level is 0..{level_max}; 0 means body (non-heading).\n"
        "The array MUST be a full permutation of the provided node_id list (no missing/extra/duplicates).\n"
        "Do NOT include any commentary, Markdown, or extra keys.\n\n"
        "Payload follows on next line as JSON. Use it strictly.\n"
        + compact
    )


"""
TOC utilities
=============

Lightweight helpers to derive a heading/section outline from DocTree nodes.
These utilities are adapted from the utils_2 prototype and are used to inject
document structure hints into Router/Planner prompts.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional


def format_heading_outline(
    flat_root: Dict,
    *,
    by_logical_page: bool = False,
    include_meta: bool = True,
    max_len: int = 80,
    include_media: bool = True,
) -> List[str]:
    """
    Build a one-line outline for each heading/media node in the DocTree.

    - Indentation reflects node_level (2 spaces per level - 1).
    - Shows node_idx, logical/physical page, and optional metadata flags.
    """
    if not isinstance(flat_root, dict):
        return []

    lines: List[str] = []

    def _walk(node: Dict) -> None:
        if not isinstance(node, dict):
            return
        node_type = (node.get("type") or node.get("role") or "").lower()
        level = _resolve_level(node)
        # Skip nodes without levels or before level 1
        if level >= 1:
            if node_type in {"image", "figure", "table"} and not include_media:
                pass
            else:
                indent = "  " * max(0, level - 1)
                idx = node.get("node_idx")
                logical_page = node.get("logical_page")
                page = node.get("page_idx")
                label = _select_label(node, node_type)
                text = _shorten_text(label, max_len=max_len)

                flags: List[str] = []
                heading_meta = node.get("heading_meta")
                if include_meta and isinstance(heading_meta, dict):
                    if heading_meta.get("via"):
                        flags.append(str(heading_meta.get("via")))
                    if heading_meta.get("frozen"):
                        flags.append("frozen")
                    if heading_meta.get("inserted"):
                        flags.append("inserted")
                    if heading_meta.get("corrected_cross_level"):
                        flags.append("corrected")
                if node_type in {"image", "figure"}:
                    flags.append("image")
                elif node_type == "table":
                    flags.append("table")

                where = f"lp={logical_page}" if by_logical_page else f"p={page}"
                flag_str = f" [{','.join(flags)}]" if flags else ""
                lines.append(f"{indent}- L{level} idx={idx} {where}: {text}{flag_str}")

        for child in node.get("children", []) or []:
            _walk(child)

    for child in flat_root.get("children", []) or []:
        _walk(child)

    return lines


def print_heading_outline(
    flat_root: Dict,
    *,
    by_logical_page: bool = True,
    include_meta: bool = True,
    max_len: int = 80,
    include_media: bool = True,
) -> None:
    """Convenience helper to print the heading outline for debugging."""
    for line in format_heading_outline(
        flat_root,
        by_logical_page=by_logical_page,
        include_meta=include_meta,
        max_len=max_len,
        include_media=include_media,
    ):
        print(line)


def _shorten_text(text: Optional[str], *, max_len: int) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max(1, max_len - 3)] + "..."


def _media_title(node: Dict) -> str:
    label = node.get("label") or node.get("title") or node.get("text") or ""
    if not label:
        caption = None
        for child in node.get("children", []) or []:
            if isinstance(child, dict) and child.get("type") in {"caption", "text"}:
                caption = child.get("text")
                break
        label = caption or node.get("node_id") or ""
    return str(label)


def _resolve_level(node: Dict) -> int:
    for key in ("node_level", "level", "heading_level"):
        if key in node:
            try:
                lvl = int(node.get(key))
                if lvl >= 0:
                    return lvl
            except Exception:
                continue
    # As fallback, try to infer from path depth
    text = node.get("heading_path") or node.get("path")
    if isinstance(text, str) and text:
        return max(1, text.count(">") + 1)
    return -1


def _select_label(node: Dict, node_type: str) -> str:
    if node_type in {"image", "figure", "table"}:
        return _media_title(node)
    for key in ("text", "title", "heading_text"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value
    caption = _extract_caption(node)
    if caption:
        return caption
    node_id = node.get("node_id")
    if isinstance(node_id, str):
        return node_id
    return ""


def _extract_caption(node: Dict) -> Optional[str]:
    for child in node.get("children", []) or []:
        if isinstance(child, dict) and child.get("type") in {"caption", "text"}:
            text = child.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
    caption = node.get("caption")
    if isinstance(caption, str) and caption.strip():
        return caption.strip()
    return None


__all__ = ["format_heading_outline", "print_heading_outline"]

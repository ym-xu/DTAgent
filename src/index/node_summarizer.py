"""
Node summarization over DocTree for M2.

Generates summary.json with compact summaries for key nodes (sections, images, tables)
and prepares embed texts for index building.

Design goals:
- Dependency-light and runnable: heuristic summarizer by default
- Pluggable: can be swapped for LLM-based summarizer later
- Robust to slightly different doctree shapes
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from typing import Any, Dict, Iterable, List, Tuple, Optional

import re
import math

STOPWORDS = set(
    """
    the a an and or for to of in on at as by from with without is are was were be being been
    this that those these there here it its itself we our you your they their he she his her
    into about across within between over under above below into onto against among per via
    can could should would may might will do does did done have has had than then also not
    if else when where while which who whom whose what why how more most less least many
    much other another such same different one two three four five six seven eight nine ten
    figure table section page caption image chart plot diagram data results discussion axis axes
    """.split()
)

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]+")
TOKEN_ALL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-\.%/]*")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# 关键词白名单（优先保留），面向图表轴/系列/单位等英文术语
WHITELIST = set(
    [
        "x-axis", "y-axis", "z-axis", "axis", "axes", "xlabel", "ylabel", "legend", "series",
        "unit", "units", "percentage", "percent", "rate", "value", "values", "time",
        "year", "month", "day", "quarter", "category", "categories", "count", "mean",
        "median", "std", "min", "max"
    ]
)

# ---------- Aliases & units canonicalization ----------
_UNIT_NORMS: Dict[str, str] = {
    "%": "%", "percent": "%", "pct": "%",
    "ms": "ms", "s": "s", "sec": "s", "second": "s", "seconds": "s",
    "kbps": "kbps", "kb/s": "kbps", "mbps": "mbps", "mb/s": "mbps", "gbps": "gbps", "gb/s": "gbps", "bps": "bps",
    "hz": "hz", "khz": "khz", "mhz": "mhz", "ghz": "ghz",
    "$": "usd", "usd": "usd", "eur": "eur", "€": "eur",
    "°c": "c", "℃": "c",
}

def _normalize_unit_token(u: str) -> Optional[str]:
    u = (u or "").strip().lower()
    return _UNIT_NORMS.get(u)

def _extract_units_set(*texts: str) -> List[str]:
    units: Dict[str, bool] = {}
    pat = re.compile(r"(\d[\d\.,]*\s*(%|ms|s|kbps|mbps|gbps|bps|hz|khz|mhz|ghz|usd|€|\$|°c|℃))", re.I)
    for s in texts:
        for m in pat.findall(s or ""):
            tok = m[1]
            n = _normalize_unit_token(tok)
            if n:
                units[n] = True
    return sorted(units.keys())

def _build_aliases_text(*texts: str) -> Optional[str]:
    txt = " ".join(filter(None, texts or []))
    aliases: List[str] = []
    if re.search(r"\bprop\.|\bproportion\b|\bpct\b|\bpercent\b", txt, flags=re.I):
        aliases.append("percent pct")
    if re.search(r"\bacc(?:uracy)?\b", txt, flags=re.I):
        aliases.append("acc accuracy")
    if re.search(r"\bf1(?:-score)?\b", txt, flags=re.I):
        aliases.append("f1 f1-score")
    out = " ".join(aliases)
    return out or None
# ---------- Sentence segmentation + MMR summarization ----------
# Split on sentence-ending punctuation followed by whitespace, or on newlines
SENT_SPLIT_RE = re.compile(r"(?<=[\.!?])\s+|\n+", re.VERBOSE)

def _split_sentences(text: str) -> List[str]:
    s = (text or "").strip()
    if not s:
        return []
    # naive split on punctuation/newlines; keep short sentences filtered later
    parts = re.split(SENT_SPLIT_RE, s)
    out = []
    for p in parts:
        t = re.sub(r"\s+", " ", p.strip())
        if len(t) >= 20:  # drop too-short fragments
            out.append(t)
    if not out and s:
        out = [s]
    return out[:30]

def _tokens(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def _tokenize_all(text: Optional[str]) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in TOKEN_ALL_RE.findall(text)]


def _unique(seq: Iterable[str]) -> List[str]:
    seen: Dict[str, bool] = {}
    out: List[str] = []
    for item in seq:
        if not item:
            continue
        if item not in seen:
            seen[item] = True
            out.append(item)
    return out


def _extract_years(text: Optional[str]) -> List[str]:
    if not text:
        return []
    yrs = [m.group(0) for m in YEAR_RE.finditer(text)]
    return _unique(yrs)

def _idf_by_sentence(sentences: List[str]) -> Dict[str, float]:
    # sentence-level idf for unsupervised scoring
    df: Dict[str, int] = {}
    N = max(1, len(sentences))
    for s in sentences:
        seen = set(_tokens(s))
        for t in seen:
            if t in STOPWORDS:
                continue
            df[t] = df.get(t, 0) + 1
    idf: Dict[str, float] = {}
    for t, d in df.items():
        idf[t] = 1.0 + max(0.0, (math.log((1.0 + N) / (1.0 + d))))
    return idf

def _sent_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf: Dict[str, float] = {}
    for t in tokens:
        if t in STOPWORDS:
            continue
        tf[t] = tf.get(t, 0.0) + 1.0
    for t in list(tf.keys()):
        tf[t] *= idf.get(t, 1.0)
    return tf

def _dot(a: Dict[str, float], b: Dict[str, float]) -> float:
    return sum(av * b.get(k, 0.0) for k, av in a.items())

def _norm(a: Dict[str, float]) -> float:
    return math.sqrt(sum(v * v for v in a.values())) or 1.0

def _cos(a: Dict[str, float], b: Dict[str, float]) -> float:
    return _dot(a, b) / (_norm(a) * _norm(b))

def summarize_text_block(
    *,
    title: str,
    raw: str,
    budget_chars: int = 280,
    k: int = 2,
    keywords: Optional[List[str]] = None,
) -> str:
    """Sentence scoring + MMR to produce 1–3 sentence dense_text.
    - Score: TF-IDF weighted, with keyword boosts
    - MMR: lambda=0.75
    """
    ctx = " ".join([title or "", raw or ""]).strip()
    sents = _split_sentences(ctx)
    if not sents:
        return (ctx[:budget_chars]).strip()
    idf = _idf_by_sentence(sents)
    kw = set([w.lower() for w in (keywords or []) if isinstance(w, str)])
    vecs = []
    base_scores = []
    for s in sents:
        toks = _tokens(s)
        v = _sent_vector(toks, idf)
        score = sum(v.values()) / max(5.0, len(toks))
        if kw:
            # small boost if sentence contains keywords
            hit = sum(1 for t in toks if t in kw)
            score *= (1.0 + min(0.5, 0.1 * hit))
        vecs.append(v)
        base_scores.append(score)
    # MMR selection
    selected: List[int] = []
    lambda_ = 0.75
    while len(selected) < k and len(selected) < len(sents):
        best_i = None
        best_score = -1e9
        for i, s in enumerate(sents):
            if i in selected:
                continue
            rel = base_scores[i]
            div = 0.0
            if selected:
                div = max(_cos(vecs[i], vecs[j]) for j in selected)
            mmr = lambda_ * rel - (1 - lambda_) * div
            if mmr > best_score:
                best_score = mmr
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
        # stop early if budget reached
        text = " ".join([sents[i] for i in selected])
        if len(text) >= budget_chars:
            break
    out = " ".join([sents[i] for i in selected])
    if len(out) > budget_chars:
        out = out[:budget_chars].rsplit(" ", 1)[0]
    return out.strip()

# Figure/Table label extractors
FIG_LABEL_RE = re.compile(r"\b(fig(?:ure)?|图)\s*([A-Za-z]*\d+)\b", re.IGNORECASE)
TAB_LABEL_RE = re.compile(r"\b(table|表)\s*([A-Za-z]*\d+)\b", re.IGNORECASE)


def _extract_label(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract (kind,label,number) where kind in {figure, table}.
    Returns (None,None,None) if not found.
    """
    if not isinstance(text, str):
        return None, None, None
    t = text.strip()
    if not t:
        return None, None, None
    m = FIG_LABEL_RE.search(t)
    if m:
        num = m.group(2)
        label = f"Figure {num}"
        return "figure", label, num
    m = TAB_LABEL_RE.search(t)
    if m:
        num = m.group(2)
        label = f"Table {num}"
        return "table", label, num
    return None, None, None


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _text(x: Any) -> str:
    if isinstance(x, str):
        return x
    return "" if x is None else str(x)



def _clean_join(parts: Iterable[str], sep: str = " ") -> str:
    out = sep.join((p or "").strip() for p in parts if isinstance(p, str) and p.strip())
    return re.sub(r"\s+", " ", out).strip()


def _clip(s: str, max_len: int = 768) -> str:
    if not s:
        return s
    if len(s) <= max_len:
        return s
    # try to cut at a space near boundary
    cut = max_len
    sp = s.rfind(" ", max(0, max_len - 40), max_len)
    if sp != -1:
        cut = sp
    return s[:cut].rstrip()


def _hints_from_text(text: str, k: int = 6) -> List[str]:
    toks = [t.lower() for t in TOKEN_RE.findall(text)]
    # 先收集白名单
    white = [t for t in toks if t in WHITELIST]
    # 再收集常规（去停用）
    rest = [t for t in toks if t not in STOPWORDS and len(t) >= 3]
    counts = Counter(rest)
    ordered = [w for w, _c in counts.most_common(max(k, 10))]
    merged: List[str] = []
    for t in white + ordered:
        if t not in merged:
            merged.append(t)
        if len(merged) >= k:
            break
    return merged[:k]


def _gather_paragraph_snippets(node: Dict[str, Any], id2node: Dict[str, Dict[str, Any]], limit_chars: int = 400) -> str:
    # Collect first paragraph-like content from children or neighbors (best-effort)
    texts: List[str] = []
    for ch in node.get("children", []) or []:
        if not isinstance(ch, dict):
            continue
        role = ch.get("role") or ch.get("type")
        if role in ("paragraph", "text"):
            t = ch.get("text")
            if isinstance(t, str) and t.strip():
                texts.append(t.strip())
                if sum(len(s) for s in texts) >= limit_chars:
                    break
    s = _clean_join(texts)
    return s[:limit_chars]


def _gather_caption(node: Dict[str, Any]) -> str:
    for ch in node.get("children", []) or []:
        if isinstance(ch, dict) and (ch.get("role") == "caption" or ch.get("type") == "text"):
            t = ch.get("text")
            if isinstance(t, str) and t.strip():
                return t.strip()
    return ""


def summarize_section(node: Dict[str, Any], id2node: Dict[str, Dict[str, Any]]) -> Tuple[str, List[str]]:
    title = _text(node.get("title"))
    snip = _gather_paragraph_snippets(node, id2node, limit_chars=400)
    if snip:
        summary = f"{title}: {snip}"
    else:
        summary = title or "Section summary"
    hints = _hints_from_text(f"{title} {snip}", k=6)
    return summary, hints


def summarize_image(node: Dict[str, Any]) -> Tuple[str, List[str]]:
    title = _text(node.get("title"))
    desc = _text(node.get("description"))
    cap = _gather_caption(node)
    base = _clean_join([title, cap, desc])
    if not base:
        base = "Image content; describe axes/series if present."
    summary = base
    # 轴/系列等特征优先作为 hints
    hints = _hints_from_text(base, k=6)
    return summary, hints


def summarize_table(node: Dict[str, Any]) -> Tuple[str, List[str]]:
    title = _text(node.get("title"))
    cap = _gather_caption(node)
    data = node.get("data")
    data_text = _text(data)
    # 从第一行提取列名（若为HTML片段或列表结构）
    headers: List[str] = []
    try:
        if isinstance(data, str):
            # 取首个 <tr> 内的 <th>/<td>
            row_match = re.search(r"<tr[\s\S]*?</tr>", data, re.IGNORECASE)
            if row_match:
                cells = re.findall(r"<t[hd][^>]*>([\s\S]*?)</t[hd]>", row_match.group(0), re.IGNORECASE)
                for c in cells:
                    txt = re.sub(r"<[^>]+>", " ", c)
                    txt = re.sub(r"\s+", " ", txt).strip()
                    if txt:
                        headers.append(txt)
        elif isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                headers = list(first.keys())
            elif isinstance(first, (list, tuple)):
                headers = [str(x) for x in first]
    except Exception:
        pass
    base = _clean_join([title, cap])
    if not base:
        base = "Table comparing key variables; summarize columns and rows."
    sample = (data_text or "")[:200]
    if sample:
        base = f"{base}. Sample: {sample}"
    summary = base
    # hints 优先包含列名
    hint_text = " ".join(headers + [title, cap, data_text])
    hints = list(dict.fromkeys(headers + _hints_from_text(hint_text, k=6)))[:6]
    return summary, hints


def _extract_table_headers(node: Dict[str, Any]) -> List[str]:
    headers: List[str] = []
    data = node.get("data")
    try:
        if isinstance(data, str):
            row_match = re.search(r"<tr[\s\S]*?</tr>", data, re.IGNORECASE)
            if row_match:
                cells = re.findall(r"<t[hd][^>]*>([\s\S]*?)</t[hd]>", row_match.group(0), re.IGNORECASE)
                for c in cells:
                    txt = re.sub(r"<[^>]+>", " ", c)
                    txt = re.sub(r"\s+", " ", txt).strip()
                    if txt:
                        headers.append(txt)
        elif isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                headers = list(first.keys())
            elif isinstance(first, (list, tuple)):
                headers = [str(x) for x in first]
    except Exception:
        pass
    return headers


class _TableHTMLExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.in_header_scope = False
        self._done = False
        self.current_row: List[str] = []
        self.current_cell: List[str] = []
        self.current_cell_is_header = False
        self.current_row_is_header = False
        self.rows: List[List[str]] = []
        self.row_is_header: List[bool] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        tag_l = tag.lower()
        if self._done:
            return
        if tag_l == "table":
            if not self.in_table:
                self.in_table = True
            return
        if not self.in_table:
            return
        if tag_l == "thead":
            self.in_header_scope = True
        elif tag_l == "tr":
            self._start_row()
        elif tag_l in ("td", "th"):
            self._start_cell(is_header=(tag_l == "th") or self.in_header_scope)
        elif tag_l == "br" and self.in_cell:
            self.current_cell.append("\n")

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        tag_l = tag.lower()
        if tag_l == "table" and self.in_table:
            if self.in_cell:
                self._end_cell()
            if self.in_row:
                self._end_row()
            self.in_table = False
            self._done = True
            return
        if not self.in_table:
            return
        if tag_l == "thead":
            self.in_header_scope = False
        elif tag_l in ("td", "th") and self.in_cell:
            self._end_cell()
        elif tag_l == "tr" and self.in_row:
            self._end_row()

    def handle_startendtag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if tag.lower() == "br" and self.in_cell:
            self.current_cell.append("\n")

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self.in_cell and data:
            self.current_cell.append(data)

    def _start_row(self) -> None:
        if self.in_row:
            return
        self.in_row = True
        self.current_row = []
        self.current_row_is_header = False

    def _end_row(self) -> None:
        self.rows.append(self.current_row)
        self.row_is_header.append(self.current_row_is_header)
        self.in_row = False
        self.current_row = []
        self.current_row_is_header = False

    def _start_cell(self, *, is_header: bool) -> None:
        self.in_cell = True
        self.current_cell = []
        self.current_cell_is_header = is_header

    def _end_cell(self) -> None:
        text = self._normalize_cell_text("".join(self.current_cell))
        self.current_row.append(text)
        if self.current_cell_is_header:
            self.current_row_is_header = True
        self.in_cell = False
        self.current_cell = []
        self.current_cell_is_header = False

    @staticmethod
    def _normalize_cell_text(text: str) -> str:
        if not text:
            return ""
        text = unescape(text)
        text = text.replace("\r", "")
        parts = []
        for line in text.split("\n"):
            line = re.sub(r"\s+", " ", line.strip())
            if line:
                parts.append(line)
        return "<br>".join(parts)


def _coerce_row_length(row: List[str], *, num_cols: int) -> List[str]:
    if len(row) >= num_cols:
        return row[:num_cols]
    return row + [""] * (num_cols - len(row))


def _normalize_header_names(headers: List[str]) -> List[str]:
    if not headers:
        return headers
    seen: Dict[str, int] = {}
    norm_headers: List[str] = []
    for idx, h in enumerate(headers):
        name = h.strip() if h else f"col{idx + 1}"
        if not name:
            name = f"col{idx + 1}"
        count = seen.get(name.lower(), 0)
        if count:
            name = f"{name}_{count+1}"
        seen[name.lower()] = count + 1
        norm_headers.append(name)
    return norm_headers


def _extract_table_struct(node: Dict[str, Any]) -> Tuple[List[str], List[List[str]]]:
    data = node.get("data")
    headers: List[str] = []
    rows: List[List[str]] = []
    header_idx = 0
    try:
        if isinstance(data, str):
            if "<table" in data.lower():
                parser = _TableHTMLExtractor()
                parser.feed(data)
                parser.close()
                if parser.rows:
                    header_idx = next((i for i, is_h in enumerate(parser.row_is_header) if is_h), 0)
                    headers = parser.rows[header_idx] if parser.rows else []
                    rows = [r for i, r in enumerate(parser.rows) if i != header_idx]
            else:
                # treat as delimited text fallback
                lines = [line.strip() for line in data.splitlines() if line.strip()]
                if lines:
                    parts = [re.split(r"\s*\|\s*", ln) for ln in lines]
                    headers = parts[0]
                    rows = parts[1:]
        elif isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                headers = list(first.keys())
                for record in data:
                    if isinstance(record, dict):
                        rows.append([str(record.get(h, "")) for h in headers])
            elif isinstance(first, (list, tuple)):
                headers = [str(x) for x in first]
                for record in data[1:]:
                    if isinstance(record, (list, tuple)):
                        rows.append([str(x) for x in record])
        # fallback: explicit rows field
        if not rows and isinstance(node.get("rows"), list):
            row_items = node.get("rows")
            if row_items:
                first = row_items[0]
                if isinstance(first, dict):
                    if not headers:
                        headers = list(first.keys())
                    for record in row_items:
                        if isinstance(record, dict):
                            rows.append([str(record.get(h, "")) for h in headers])
                elif isinstance(first, (list, tuple)):
                    if not headers:
                        headers = [f"col{i+1}" for i in range(len(first))]
                    for record in row_items:
                        if isinstance(record, (list, tuple)):
                            rows.append([str(x) for x in record])
    except Exception:
        headers, rows = [], []

    headers = _normalize_header_names(headers)
    if headers and rows:
        num_cols = len(headers)
        rows = [_coerce_row_length([str(c) for c in row], num_cols=num_cols) for row in rows]
    elif rows:
        num_cols = max(len(row) for row in rows)
        headers = [f"col{i+1}" for i in range(num_cols)]
        rows = [_coerce_row_length([str(c) for c in row], num_cols=num_cols) for row in rows]
    else:
        headers = []
        rows = []
    return headers, rows


def _extract_unit_from_header(header: str) -> Optional[str]:
    if not header:
        return None
    m = re.search(r"\(([^)]+)\)", header)
    if not m:
        return None
    return _normalize_unit_token(m.group(1))


def _infer_year_from_values(*values: str) -> Optional[str]:
    for val in values:
        for year in _extract_years(val):
            return year
    return None


def _build_table_row_records(
    table_node: Dict[str, Any],
    headers: List[str],
    rows: List[List[str]],
    *,
    parent_section: Optional[str],
    parent_title: Optional[str],
) -> List[Dict[str, Any]]:
    if not headers or not rows:
        return []
    nid = table_node.get("node_id")
    if not isinstance(nid, str):
        return []
    page_idx = table_node.get("page_idx")
    label = table_node.get("label")
    chart_type = table_node.get("kind")
    records: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        row_label = row[0] if row else ""
        columns: List[Dict[str, Any]] = []
        token_bag: List[str] = _tokenize_all(row_label)
        for cidx, header in enumerate(headers):
            value = row[cidx] if cidx < len(row) else ""
            unit_from_header = _extract_unit_from_header(header)
            unit_from_value = None
            units = _extract_units_set(value)
            if units:
                unit_from_value = units[0]
            unit_final = unit_from_header or unit_from_value
            year = _infer_year_from_values(header, value)
            col_tokens = _tokenize_all(header) + _tokenize_all(value)
            token_bag.extend(col_tokens)
            if unit_final:
                token_bag.append(unit_final)
            col_entry: Dict[str, Any] = {
                "name": header,
                "index": cidx,
                "value": value,
            }
            if unit_final:
                col_entry["unit"] = unit_final
            if year:
                try:
                    col_entry["year"] = int(year)
                except Exception:
                    col_entry["year"] = year
            columns.append(col_entry)
        dense_text = "; ".join([f"{headers[i]}={row[i]}" for i in range(len(headers))])
        if row_label and row_label not in dense_text:
            dense_text = f"{row_label}: {dense_text}"
        record = {
            "node_id": nid,
            "row_id": f"{nid}:row:{idx}",
            "role": "table_row",
            "row_index": idx,
            "row_label": row_label or None,
            "columns": columns,
            "dense_text": dense_text,
            "normalized_tokens": _unique(token_bag),
            "filters": {
                **({"page_idx": page_idx} if page_idx is not None else {}),
                **({"parent_section": parent_section} if parent_section else {}),
                **({"parent_title": parent_title} if parent_title else {}),
                **({"label": label} if label else {}),
                **({"chart_type": chart_type} if chart_type else {}),
            },
        }
        records.append(record)
    return records


def _extract_entities(text: Optional[str]) -> List[str]:
    if not text:
        return []
    # Simple heuristic: capitalized word sequences
    entities = re.findall(r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*)\b", text)
    return _unique([e.strip() for e in entities])


def _build_figure_spans(
    node: Dict[str, Any],
    *,
    parent_section: Optional[str],
    parent_title: Optional[str],
) -> List[Dict[str, Any]]:
    nid = node.get("node_id")
    if not isinstance(nid, str):
        return []
    page_idx = node.get("page_idx")
    chart_type = (node.get("kind") or "").lower() or None
    spans: List[Dict[str, Any]] = []

    def _add_span(span_role: str, text: Optional[str], *, source: str) -> None:
        if not text:
            return
        text = text.strip()
        if not text:
            return
        tokens = _tokenize_all(text)
        span_id = f"{nid}:{span_role}:{len(spans)}"
        span = {
            "node_id": nid,
            "span_id": span_id,
            "role": span_role,
            "chart_type": chart_type,
            "dense_text": text,
            "sparse_tokens": tokens,
            "entities": _extract_entities(text),
            "units_set": _extract_units_set(text),
            "years": _extract_years(text),
            "source": source,
            "filters": {
                **({"page_idx": page_idx} if page_idx is not None else {}),
                **({"parent_section": parent_section} if parent_section else {}),
                **({"parent_title": parent_title} if parent_title else {}),
            },
        }
        spans.append(span)

    # caption
    cap = _gather_caption(node)
    _add_span("figure_caption", cap, source="caption")
    # description
    desc = _text(node.get("description"))
    _add_span("figure_description", desc, source="description")
    # legend items
    legend_items = node.get("legend") or node.get("legend_items")
    if isinstance(legend_items, str):
        legend_items = [legend_items]
    if isinstance(legend_items, list):
        for item in legend_items:
            if isinstance(item, str):
                _add_span("figure_legend", item, source="legend")
            elif isinstance(item, dict):
                text = item.get("text") or item.get("label")
                _add_span("figure_legend", _text(text), source="legend")
    # axis labels
    axis = node.get("axis_labels") or node.get("axes")
    if isinstance(axis, dict):
        for key in ("x", "y", "z", "x_axis", "y_axis", "z_axis"):
            if key in axis:
                _add_span("axis_label", _text(axis[key]), source=key)
    elif isinstance(axis, list):
        for item in axis:
            if isinstance(item, str):
                _add_span("axis_label", item, source="axis")
            elif isinstance(item, dict):
                _add_span("axis_label", _text(item.get("text")), source=item.get("axis") or "axis")
    # OCR text
    ocr = node.get("ocr_text")
    if isinstance(ocr, str):
        # split into sentences to avoid giant chunks
        for chunk in re.split(r"[;\n]+", ocr):
            _add_span("figure_ocr", chunk, source="ocr")
    elif isinstance(ocr, list):
        for entry in ocr:
            if isinstance(entry, str):
                _add_span("figure_ocr", entry, source="ocr")
            elif isinstance(entry, dict):
                _add_span("figure_ocr", _text(entry.get("text")), source="ocr")

    return spans


def build_id_index(root: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    id2node: Dict[str, Dict[str, Any]] = {}
    stack = [root]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict) and isinstance(cur.get("node_id"), str):
            id2node[cur["node_id"]] = cur
        for ch in cur.get("children", []) or []:
            if isinstance(ch, dict):
                stack.append(ch)
    return id2node


def build_parent_section_map(root: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Map each node_id to its nearest parent section id (or None)."""
    parent: Dict[str, Optional[str]] = {}
    stack: List[Tuple[Dict[str, Any], Optional[str]]] = [(root, None)]
    while stack:
        cur, cur_sec = stack.pop()
        if not isinstance(cur, dict):
            continue
        nid = cur.get("node_id")
        t = cur.get("type") or cur.get("role")
        sec = cur_sec
        if t == "section" and isinstance(nid, str):
            sec = nid
        if isinstance(nid, str):
            parent[nid] = cur_sec
        for ch in cur.get("children", []) or []:
            if isinstance(ch, dict):
                stack.append((ch, sec))
    return parent


def _ancestor_titles(nid: str, parent_map: Dict[str, Optional[str]], id2node: Dict[str, Dict[str, Any]]) -> List[str]:
    titles: List[str] = []
    cur = parent_map.get(nid)
    visited = set()
    while cur and cur not in visited:
        visited.add(cur)
        node = id2node.get(cur)
        t = _text(node.get("title") if isinstance(node, dict) else None)
        if t:
            titles.append(t)
        cur = parent_map.get(cur)
    titles.reverse()
    return titles


def _aggregate_section_subtree(sec_node: Dict[str, Any], id2node: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate subtree signals for a section: page_span, labels, schemas, captions, chart_types, units, stats, representatives."""
    pages: List[int] = []
    labels: List[str] = []
    schemas: List[str] = []
    captions: List[str] = []
    chart_types: Dict[str, bool] = {}
    units: Dict[str, bool] = {}
    stats = {"sections": 0, "images": 0, "tables": 0, "equations": 0, "paragraphs": 0}
    reps: Dict[str, Optional[str]] = {"top_image": None, "top_table": None, "top_paragraph": None}

    # immediate child section headings for child_surface
    child_headings: List[str] = []

    stack = [sec_node]
    seen = set()
    while stack:
        cur = stack.pop()
        if not isinstance(cur, dict) or id(cur) in seen:
            continue
        seen.add(id(cur))
        t = cur.get("type") or cur.get("role")
        pid = cur.get("page_idx")
        if isinstance(pid, int):
            pages.append(pid)
        if t == "section":
            stats["sections"] += 1
        elif t == "image":
            stats["images"] += 1
            cap = _gather_caption(cur)
            if cap:
                captions.append(cap)
                for u in _extract_units_set(cap):
                    units[u] = True
            kind = (cur.get("kind") or "").lower()
            if kind:
                chart_types[kind] = True
            # label
            _k, lbl, _num = _extract_label(_clean_join([_text(cur.get("title")), cap]))
            if lbl:
                labels.append(lbl)
            if reps["top_image"] is None and isinstance(cur.get("node_id"), str):
                reps["top_image"] = cur["node_id"]
        elif t == "table":
            stats["tables"] += 1
            cap = _gather_caption(cur)
            if cap:
                captions.append(cap)
                for u in _extract_units_set(cap):
                    units[u] = True
            hdrs = _extract_table_headers(cur)
            if hdrs:
                schemas.extend(hdrs)
                # parse units within headers
                for h in hdrs:
                    m = re.search(r"\(([^)]+)\)", str(h))
                    if m:
                        u = _normalize_unit_token(m.group(1))
                        if u:
                            units[u] = True
            _k, lbl, _num = _extract_label(_clean_join([_text(cur.get("title")), cap]))
            if lbl:
                labels.append(lbl)
            if reps["top_table"] is None and isinstance(cur.get("node_id"), str):
                reps["top_table"] = cur["node_id"]
        elif t in ("text", "paragraph"):
            stats["paragraphs"] += 1
            txt = _text(cur.get("text"))
            for u in _extract_units_set(txt):
                units[u] = True
            if reps["top_paragraph"] is None and isinstance(cur.get("node_id"), str):
                reps["top_paragraph"] = cur["node_id"]
        elif t == "equation":
            stats["equations"] += 1

        # collect immediate child section headings
        for ch in cur.get("children", []) or []:
            if isinstance(ch, dict):
                if cur is sec_node and (ch.get("type") == "section"):
                    ht = _text(ch.get("title"))
                    if ht:
                        child_headings.append(ht)
                stack.append(ch)

    page_span = [min(pages), max(pages)] if pages else None
    uniq = lambda xs: list(dict.fromkeys([x for x in xs if x]))
    labels_u = uniq(labels)
    schemas_u = uniq(schemas)
    captions_u = uniq(captions)
    chart_types_u = sorted(chart_types.keys())
    units_u = sorted(units.keys())
    # child surface text basis
    child_surface_basis = "; ".join(filter(None, ["; ".join(child_headings[:8]), "; ".join(captions_u[:12]), "; ".join(schemas_u[:12])]))
    keywords_topk = _hints_from_text(child_surface_basis, k=6)
    return {
        "page_span": page_span,
        "child_labels": labels_u,
        "child_table_schema": schemas_u,
        "caption_bag": "; ".join(captions_u[:20]),
        "chart_types": chart_types_u,
        "units_set": units_u,
        "subtree_stats": stats,
        "representatives": reps,
        "child_headings": child_headings,
        "keywords_topk": keywords_topk,
        "child_surface_basis": child_surface_basis,
    }


## iter_nodes is unused; prefer iter_nodes_with_section


def iter_nodes_with_section(root: Dict[str, Any]) -> Iterable[Tuple[Dict[str, Any], Optional[str]]]:
    """DFS yield (node, nearest_parent_section_id)."""
    stack: List[Tuple[Dict[str, Any], Optional[str]]] = [(root, None)]
    while stack:
        cur, cur_sec = stack.pop()
        if not isinstance(cur, dict):
            continue
        t = cur.get("type") or cur.get("role")
        sec = cur_sec
        if t == "section" and isinstance(cur.get("node_id"), str):
            sec = cur.get("node_id")
        yield cur, sec
        for ch in reversed(cur.get("children", []) or []):
            if isinstance(ch, dict):
                stack.append((ch, sec))


def should_summarize(node: Dict[str, Any], include_leaves: bool = False) -> bool:
    t = node.get("type") or node.get("role")
    if t == "section":
        return True
    if t == "image":
        # Heuristic kinds
        k = (node.get("kind") or "").lower()
        desc = _text(node.get("description"))
        if k in {"statistical", "diagram"}:
            return True
        if any(w in desc.lower() for w in ["chart", "plot", "graph", "diagram"]):
            return True
        return False
    if t == "table":
        return True
    if include_leaves and t in ("text", "paragraph", "list", "equation"):
        return True
    return False


def _llm_call(
    messages: List[Dict[str, Any]],
    *,
    backend: str = "auto",
    model: str = "gpt-4o-mini",
    json_mode: bool = True,
) -> Optional[str]:
    try:
        if backend == "qwen" or (backend == "auto" and ("qwen" in model.lower())):
            from ..utils.llm_clients import qwen_llm_call  # type: ignore
            return qwen_llm_call(messages, images=None, model=model, json_mode=json_mode)
        else:
            from ..utils.llm_clients import gpt_llm_call  # type: ignore
            return gpt_llm_call(messages, images=None, model=model, json_mode=json_mode)
    except Exception:
        return None


def summarize_with_llm(
    *,
    node: Dict[str, Any],
    node_type: str,
    title: str,
    page_idx: Any,
    id2node: Dict[str, Dict[str, Any]],
    llm_backend: str,
    llm_model: str,
) -> Optional[Tuple[str, List[str]]]:
    """Use an LLM to produce {summary, hints} for a node. Returns None on failure."""
    # Gather context per type
    if node_type == "section":
        snip = _gather_paragraph_snippets(node, id2node, limit_chars=600)
        ctx = {
            "type": node_type,
            "title": title,
            "page_idx": page_idx,
            "text": snip,
        }
        hint_note = "Focus on core concepts, variables, datasets, important findings."
    elif node_type == "image":
        cap = _gather_caption(node)
        desc = _text(node.get("description"))
        kind = node.get("kind")
        ctx = {
            "type": node_type,
            "title": title,
            "page_idx": page_idx,
            "caption": cap,
            "description": desc,
            "kind": kind,
        }
        hint_note = "Prefer x-axis, y-axis, series names, units, legend labels."
    elif node_type == "table":
        data = node.get("data")
        data_text = _text(data)[:800]
        ctx = {
            "type": node_type,
            "title": title,
            "page_idx": page_idx,
            "data_preview": data_text,
        }
        hint_note = "Extract column names and measurement units if present."
    else:  # leaves
        content = _text(node.get("text"))[:800]
        ctx = {
            "type": node_type,
            "title": title,
            "page_idx": page_idx,
            "text": content,
        }
        hint_note = "Use key terms, defined concepts, formulas, variables."

    sys = {
        "role": "system",
        "content": (
            "You are a document summarizer for DocTree nodes. Output strict JSON with fields "
            "{summary: string, hints: string[]} in English. The summary should be 2-3 sentences. "
            "The hints should be 3-8 concise tokens for semantic navigation. "
            + hint_note
        ),
    }
    user = {"role": "user", "content": json.dumps(ctx, ensure_ascii=False)}
    raw = _llm_call([sys, user], backend=llm_backend, model=llm_model, json_mode=True)
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        summ = _text(obj.get("summary"))
        hints = obj.get("hints")
        if isinstance(hints, list):
            hints = [str(x) for x in hints][:8]
        else:
            hints = _hints_from_text(_text(hints), k=6)
        if not summ:
            return None
        return summ, hints  # type: ignore
    except Exception:
        return None


def role_of(node: Dict[str, Any]) -> str:
    t = _text(node.get("type")) or _text(node.get("role"))
    return t or "node"


def build_summary(
    *,
    doctree_path: str,
    out_dir: str,
    doc_id: str | None = None,
    model: str = "heuristic-v1",
    max_tokens: int = 120,
    include_leaves: bool = False,
    use_llm: bool = False,
    llm_backend: str = "auto",
    llm_model: str = "gpt-4o-mini",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Create summary.json-compatible object and embed entries from a doctree.

    Returns (summary_obj, coarse_entries, leaf_entries).
    """
    root = json.load(open(doctree_path, "r"))
    doc_id = doc_id or root.get("doc_id") or os.path.basename(os.path.dirname(doctree_path))

    id2node = build_id_index(root)
    parent_map = build_parent_section_map(root)
    nodes_out_arr: List[Dict[str, Any]] = []
    # legacy containers (not used for new index, kept to satisfy return shape)
    coarse_entries: List[Dict[str, Any]] = []
    # leaf chunk intermediate entries (no embed_text)
    leaf_entries: List[Dict[str, Any]] = []
    # new views accumulators
    dense_coarse: List[Dict[str, Any]] = []
    dense_leaf: List[Dict[str, Any]] = []
    sparse_coarse: List[Dict[str, Any]] = []
    sparse_leaf: List[Dict[str, Any]] = []
    table_cells: List[Dict[str, Any]] = []
    figure_spans: List[Dict[str, Any]] = []
    graph_edges: List[Dict[str, Any]] = []
    id_maps: Dict[str, Any] = {"label2id": {}, "figure": {}, "table": {}}
    heading_titles: Dict[str, str] = {}
    heading_paths_map: Dict[str, List[str]] = {}
    heading_children: Dict[str, List[str]] = {}
    heading_keyword_map: Dict[str, List[str]] = {}

    def _chunk_text(s: str, limit: int = 600, overlap: int = 100) -> List[Tuple[str, int]]:
        s = re.sub(r"\s+", " ", (s or "").strip())
        if not s:
            return []
        chunks: List[Tuple[str, int]] = []
        start = 0
        n = len(s)
        while start < n:
            end = min(start + limit, n)
            if end < n:
                sp = s.rfind(" ", start + int(0.6 * limit), end)
                if sp != -1 and sp > start:
                    end = sp
            chunks.append((s[start:end].strip(), len(chunks)))
            if end >= n:
                break
            start = max(end - overlap, 0)
        return [(c, i) for (c, i) in chunks if c]

    for n, parent_sec in iter_nodes_with_section(root):
        t = n.get("type") or n.get("role")
        allowed = {"section", "image", "table", "text", "paragraph", "list", "equation"}
        if t not in allowed:
            continue
        if not should_summarize(n, include_leaves=include_leaves):
            continue
        nid = n.get("node_id")
        if not isinstance(nid, str):
            continue
        page_idx = n.get("page_idx")
        title = _text(n.get("title"))
        role = role_of(n)
        level = n.get("level") if isinstance(n.get("level"), int) else None
        bbox = n.get("bbox")

        # Resolve parent section title if any
        parent_title: Optional[str] = None
        if parent_sec and parent_sec in id2node:
            pt = id2node[parent_sec].get("title")
            if isinstance(pt, str) and pt.strip():
                parent_title = pt.strip()

        if t == "section":
            if use_llm:
                llm_res = summarize_with_llm(
                    node=n,
                    node_type="section",
                    title=title,
                    page_idx=page_idx,
                    id2node=id2node,
                    llm_backend=llm_backend,
                    llm_model=llm_model,
                )
                if llm_res:
                    summary, keywords = llm_res
                else:
                    summary, keywords = summarize_section(n, id2node)
            else:
                summary, keywords = summarize_section(n, id2node)
            kids = [ch.get("node_id") for ch in n.get("children", []) or [] if isinstance(ch, dict) and isinstance(ch.get("node_id"), str)]
            # Build dense_text via MMR
            dense_text = summarize_text_block(
                title=title,
                raw=_gather_paragraph_snippets(n, id2node, limit_chars=800),
                budget_chars=360,
                k=3,
                keywords=keywords,
            )
            node_obj = {
                "role": "section",
                **({"level": level} if level is not None else {}),
                **({"title": title} if title else {}),
                **({"page_idx": page_idx} if page_idx is not None else {}),
                "dense_text": dense_text,
                "hints": keywords,
                **({"children": kids} if kids else {}),
            }
        elif t == "image":
            if use_llm:
                llm_res = summarize_with_llm(
                    node=n,
                    node_type="image",
                    title=title,
                    page_idx=page_idx,
                    id2node=id2node,
                    llm_backend=llm_backend,
                    llm_model=llm_model,
                )
                if llm_res:
                    summary, keywords = llm_res
                else:
                    summary, keywords = summarize_image(n)
            else:
                summary, keywords = summarize_image(n)
            kind = (n.get("kind") or (
                "statistical" if any(w in _text(n.get("description")).lower() for w in ["chart", "plot", "graph"]) else "figure"
            )).lower()
            # Parse label from title or caption
            lkind, label, num = _extract_label(_clean_join([title, _gather_caption(n)]))
            if label and lkind == "figure":
                # Put Figure X into hints
                low = [h.lower() for h in (keywords or [])]
                if label.lower() not in low:
                    keywords = list(dict.fromkeys([label] + (keywords or [])))[:8]
            dense_text = summarize_text_block(
                title=title,
                raw=_clean_join([_gather_caption(n), _text(n.get("description"))])[:800],
                budget_chars=240,
                k=2,
                keywords=keywords + ([label] if label else []),
            )
            node_obj = {
                "role": "image",
                **({"kind": kind} if kind else {}),
                **({"title": title} if title else {}),
                **({"page_idx": page_idx} if page_idx is not None else {}),
                "dense_text": dense_text,
                "hints": keywords,
                **({"label": label} if label else {}),
                **({"figure_no": num} if (label and lkind == "figure") else {}),
                **({"parent_section": parent_sec} if parent_sec else {}),
                **({"parent_title": parent_title} if parent_title else {}),
            }
            if label:
                id_maps["label2id"][label] = nid
                if num:
                    id_maps["figure"][str(num)] = nid
        elif t == "table":  # table
            if use_llm:
                llm_res = summarize_with_llm(
                    node=n,
                    node_type="table",
                    title=title,
                    page_idx=page_idx,
                    id2node=id2node,
                    llm_backend=llm_backend,
                    llm_model=llm_model,
                )
                if llm_res:
                    summary, keywords = llm_res
                else:
                    summary, keywords = summarize_table(n)
            else:
                summary, keywords = summarize_table(n)
            # Parse label
            lkind, label, num = _extract_label(_clean_join([title, _gather_caption(n)]))
            if label and lkind == "table":
                low = [h.lower() for h in (keywords or [])]
                if label.lower() not in low:
                    keywords = list(dict.fromkeys([label] + (keywords or [])))[:8]
            dense_text = summarize_text_block(
                title=title,
                raw=_clean_join([_gather_caption(n), _text(n.get("data"))])[:800],
                budget_chars=240,
                k=2,
                keywords=keywords + ([label] if label else []),
            )
            node_obj = {
                "role": "table",
                **({"title": title} if title else {}),
                **({"page_idx": page_idx} if page_idx is not None else {}),
                "dense_text": dense_text,
                "hints": keywords,
                **({"label": label} if label else {}),
                **({"table_no": num} if (label and lkind == "table") else {}),
                **({"parent_section": parent_sec} if parent_sec else {}),
                **({"parent_title": parent_title} if parent_title else {}),
            }
            if label:
                id_maps["label2id"][label] = nid
                if num:
                    id_maps["table"][str(num)] = nid
        elif t in ("text", "paragraph") and include_leaves:
            content = _text(n.get("text"))
            dense_text = summarize_text_block(
                title=title,
                raw=content[:1000],
                budget_chars=200,
                k=2,
                keywords=_hints_from_text(content, k=6),
            )
            summary, keywords = dense_text or ("Paragraph"), _hints_from_text(content, k=6)
            node_obj = {
                "role": "paragraph",
                **({"page_idx": page_idx} if page_idx is not None else {}),
                "dense_text": summary,
                "hints": keywords,
            }
            # 叶子索引分片
            for chunk, ci in _chunk_text(content, limit=600, overlap=80):
                ch_hints = _hints_from_text(chunk, k=6)
                # also inject Figure/Table labels found in text
                lkind, label, num = _extract_label(chunk)
                if label:
                    low = [h.lower() for h in ch_hints]
                    if label.lower() not in low:
                        ch_hints = list(dict.fromkeys([label] + ch_hints))[:8]
                leaf_entries.append({
                    "node_id": f"{nid}#c{ci}",
                    "orig_node_id": nid,
                    "chunk_idx": ci,
                    "role": "paragraph",
                    "page_idx": page_idx,
                    "raw_text": chunk,
                    "hints": ch_hints,
                    **({"label": label} if label else {}),
                    **({"figure_no": num} if (label and lkind == "figure") else {}),
                    **({"table_no": num} if (label and lkind == "table") else {}),
                    **({"parent_section": parent_sec} if parent_sec else {}),
                    **({"bbox": bbox} if bbox is not None else {}),
                })
        elif t == "list" and include_leaves:
            items = n.get("items") if isinstance(n.get("items"), list) else []
            head = "; ".join(str(x.get("text") if isinstance(x, dict) else x) for x in items[:5])
            summary, keywords = (f"List: {head}" if head else "List"), _hints_from_text(head, k=6)
            node_obj = {
                "role": "list",
                **({"page_idx": page_idx} if page_idx is not None else {}),
                "dense_text": summary,
                "hints": keywords,
            }
            full = "; ".join(str(x.get("text") if isinstance(x, dict) else x) for x in items)
            for chunk, ci in _chunk_text(full, limit=600, overlap=80):
                ch_hints = _hints_from_text(chunk, k=6)
                lkind, label, num = _extract_label(chunk)
                if label:
                    low = [h.lower() for h in ch_hints]
                    if label.lower() not in low:
                        ch_hints = list(dict.fromkeys([label] + ch_hints))[:8]
                leaf_entries.append({
                    "node_id": f"{nid}#c{ci}",
                    "orig_node_id": nid,
                    "chunk_idx": ci,
                    "role": "list",
                    "page_idx": page_idx,
                    "raw_text": chunk,
                    "hints": ch_hints,
                    **({"label": label} if label else {}),
                    **({"figure_no": num} if (label and lkind == "figure") else {}),
                    **({"table_no": num} if (label and lkind == "table") else {}),
                    **({"parent_section": parent_sec} if parent_sec else {}),
                    **({"bbox": bbox} if bbox is not None else {}),
                })
        elif t == "equation" and include_leaves:
            eq = _text(n.get("text"))
            dense_text = summarize_text_block(
                title=title,
                raw=eq[:800],
                budget_chars=200,
                k=2,
                keywords=_hints_from_text(eq, k=6),
            )
            summary, keywords = (dense_text or "Equation"), _hints_from_text(eq, k=6)
            node_obj = {
                "role": "equation",
                **({"page_idx": page_idx} if page_idx is not None else {}),
                "dense_text": summary,
                "hints": keywords,
            }
            # 方程作为单条 leaf（不分片）
            leaf_entries.append({
                "node_id": nid,
                "role": "equation",
                "page_idx": page_idx,
                "raw_text": eq,
                "hints": keywords,
                **({"parent_section": parent_sec} if parent_sec else {}),
                **({"bbox": bbox} if bbox is not None else {}),
            })
        elif include_leaves and (n.get("role") == "caption"):
            cap = _text(n.get("text"))
            dense_text = summarize_text_block(
                title=title,
                raw=cap[:800],
                budget_chars=200,
                k=2,
                keywords=_hints_from_text(cap, k=6),
            )
            summary, keywords = (dense_text or "Caption"), _hints_from_text(cap, k=6)
            lkind, label, num = _extract_label(cap)
            if label:
                low = [h.lower() for h in (keywords or [])]
                if label.lower() not in low:
                    keywords = list(dict.fromkeys([label] + (keywords or [])))[:8]
            node_obj = {
                "role": "caption",
                **({"page_idx": page_idx} if page_idx is not None else {}),
                "dense_text": summary,
                "hints": keywords,
                **({"label": label} if label else {}),
                **({"figure_no": num} if (label and lkind == "figure") else {}),
                **({"table_no": num} if (label and lkind == "table") else {}),
            }
            leaf_entries.append({
                "node_id": nid,
                "orig_node_id": nid,
                "role": "caption",
                "page_idx": page_idx,
                "raw_text": cap,
                "hints": keywords,
                **({"parent_section": parent_sec} if parent_sec else {}),
                **({"bbox": bbox} if bbox is not None else {}),
                **({"label": label} if label else {}),
                **({"figure_no": num} if (label and lkind == "figure") else {}),
                **({"table_no": num} if (label and lkind == "table") else {}),
            })
        else:
            continue

        node_obj_with_id = {"node_id": nid, **node_obj}
        nodes_out_arr.append(node_obj_with_id)
        # 旧的 coarse_entries embed_text 路径已废弃（使用 dense/sparse 视图）
        # 图边：parent_section → image/table；表列边
        if t in ("image", "table") and parent_sec:
            graph_edges.append({"src": parent_sec, "dst": nid, "type": "child"})
        if t == "table":
            cols = _extract_table_headers(n)
            for col in cols:
                col_str = str(col)
                unit_norm = None
                m = re.search(r"\(([^)]+)\)", col_str)
                if m:
                    unit_norm = _normalize_unit_token(m.group(1)) or None
                graph_edges.append({
                    "src": nid,
                    "dst": f"{nid}:col:{col_str}",
                    "type": "has_col",
                    **({"unit": unit_norm} if unit_norm else {}),
                })

    summary_obj: Dict[str, Any] = {
        "doc_id": doc_id,
        "meta": {
            "model": model,
            "max_tokens": int(max_tokens),
            "ts": _now_iso(),
            "use_llm": bool(use_llm),
            "llm_model": llm_model if use_llm else None,
            "backend": llm_backend if use_llm else None,
        },
        "nodes": nodes_out_arr,
    }
    # Build dense/sparse views and graph edges/id_maps
    def _has_digits(s: str) -> bool:
        return any(ch.isdigit() for ch in s or "")

    for node in nodes_out_arr:
        nid = node.get("node_id")
        role = node.get("role")
        if role in ("section", "image", "table"):
            units_set = _extract_units_set(node.get("dense_text") or "", node.get("title") or "")
            if role == "section":
                sec_agg = _aggregate_section_subtree(id2node.get(nid) or {}, id2node)
                headings_path = _ancestor_titles(nid, parent_map, id2node)
                title = node.get("title") or ""
                gist_text = node.get("dense_text") or ""
                if title:
                    heading_titles[nid] = title
                full_path = headings_path + ([title] if title else [])
                if full_path:
                    heading_paths_map[nid] = full_path
                orig_section = id2node.get(nid) or {}
                children_ids: List[str] = []
                for ch in orig_section.get("children", []) or []:
                    if isinstance(ch, dict) and isinstance(ch.get("node_id"), str):
                        children_ids.append(ch["node_id"])
                if children_ids:
                    heading_children[nid] = children_ids
                for kw in sec_agg.get("keywords_topk") or []:
                    if kw:
                        heading_keyword_map.setdefault(kw.lower(), []).append(nid)
                for tok in _tokens(title):
                    if tok:
                        heading_keyword_map.setdefault(tok.lower(), []).append(nid)
                child_basis = sec_agg.get("child_surface_basis") or ""
                child_text = summarize_text_block(
                    title=title,
                    raw=str(child_basis)[:800],
                    budget_chars=360,
                    k=3,
                    keywords=sec_agg.get("keywords_topk") or [],
                )
                path_text = " > ".join(headings_path + [str(title)]) if headings_path or title else title

                common_filters = {
                    **({"page_idx": node.get("page_idx")} if node.get("page_idx") is not None else {}),
                    **({"level": node.get("level")} if node.get("level") is not None else {}),
                    **({"parent_section": node.get("parent_section")} if node.get("parent_section") else {}),
                    **({"parent_title": node.get("parent_title")} if node.get("parent_title") else {}),
                    **({"page_span": sec_agg.get("page_span")} if sec_agg.get("page_span") else {}),
                    **({"chart_types": sec_agg.get("chart_types")} if sec_agg.get("chart_types") else {}),
                    **({"units_set": sec_agg.get("units_set")} if sec_agg.get("units_set") else {}),
                }
                stats = sec_agg.get("subtree_stats") or {}
                aff = {
                    "has_numbers": bool(sec_agg.get("units_set")) or _has_digits(gist_text),
                    "supports_COMPARE": bool(stats.get("tables", 0) or stats.get("images", 0)),
                    "supports_LOOKUP": bool(stats.get("tables", 0)),
                    "supports_TREND": ("statistical" in (sec_agg.get("chart_types") or [])),
                    "supports_ROUTE_TO_image": bool(stats.get("images", 0)),
                    "supports_ROUTE_TO_table": bool(stats.get("tables", 0)),
                    "supports_EXPAND_CHILDREN": True,
                }
                for variant, text in (
                    ("heading", title),
                    ("gist", gist_text),
                    ("child", child_text),
                    ("path", path_text),
                ):
                    dense_coarse.append({
                        "node_id": f"{nid}#" + ("h" if variant == "heading" else ("g" if variant == "gist" else ("c" if variant == "child" else "p"))),
                        "variant": variant,
                        "role": role,
                        "dense_text": text or "",
                        "title": title,
                        "summary": gist_text or title,
                        "filters": common_filters,
                        "affordances": aff,
                        "subtree_sketch": {
                            "keywords_topk": sec_agg.get("keywords_topk"),
                            "representatives": sec_agg.get("representatives"),
                        },
                    })
                child_labels = "; ".join(sec_agg.get("child_labels") or []) or None
                child_schema = "; ".join(sec_agg.get("child_table_schema") or []) or None
                caption_bag = sec_agg.get("caption_bag") or None
                aliases_text = _build_aliases_text(child_schema or "", caption_bag or "")
                sparse_coarse.append({
                    "id": nid,
                    "role": role,
                    "heading": title or None,
                    "headings_path": " > ".join(headings_path) if headings_path else None,
                    "child_labels": child_labels,
                    "child_table_schema": child_schema,
                    "caption_bag": caption_bag,
                    "aliases": aliases_text,
                    "body": gist_text or None,
                    "filters": {
                        **({"level": node.get("level")} if node.get("level") is not None else {}),
                        **({"page_span": sec_agg.get("page_span")} if sec_agg.get("page_span") else {}),
                        **({"parent_section": node.get("parent_section")} if node.get("parent_section") else {}),
                    },
                })
            else:
                orig = dict(id2node.get(nid) or {})
                if node.get("label") and "label" not in orig:
                    orig["label"] = node.get("label")
                if node.get("kind") and "kind" not in orig:
                    orig["kind"] = node.get("kind")
                if node.get("page_idx") is not None and orig.get("page_idx") is None:
                    orig["page_idx"] = node.get("page_idx")
                dense_coarse.append({
                    "node_id": nid,
                    "role": role,
                    "dense_text": node.get("dense_text") or "",
                    "title": node.get("title"),
                    "summary": node.get("dense_text"),
                    "filters": {
                        **({"page_idx": node.get("page_idx")} if node.get("page_idx") is not None else {}),
                        **({"level": node.get("level")} if node.get("level") is not None else {}),
                        **({"parent_section": node.get("parent_section")} if node.get("parent_section") else {}),
                        **({"parent_title": node.get("parent_title")} if node.get("parent_title") else {}),
                        **({"chart_type": node.get("kind")} if node.get("kind") else {}),
                        **({"label": node.get("label")} if node.get("label") else {}),
                        **({"figure_no": node.get("figure_no")} if node.get("figure_no") else {}),
                        **({"table_no": node.get("table_no")} if node.get("table_no") else {}),
                        **({"units_set": units_set} if units_set else {}),
                    },
                    "affordances": {
                        "has_numbers": _has_digits(node.get("dense_text") or ""),
                        "supports_COMPARE": (role == "table") or (node.get("kind") == "statistical"),
                        "supports_LOOKUP": (role == "table"),
                        "supports_TREND": ("trend" in (node.get("dense_text") or "").lower()) or (node.get("kind") == "statistical"),
                    },
                })
                cap_txt = None
                schema = None
                try:
                    if role == "image" and orig:
                        cap_txt = _gather_caption(orig) or None
                    elif role == "table" and orig:
                        cap_txt = _gather_caption(orig) or None
                        headers = _extract_table_headers(orig)
                        schema = "; ".join(headers) if headers else None
                except Exception:
                    pass
                aliases_text = _build_aliases_text(node.get("title") or "", node.get("dense_text") or "")
                if role == "table":
                    aliases_text = _build_aliases_text(schema or "", node.get("title") or "")
                sparse_coarse.append({
                    "id": nid,
                    "role": role,
                    "title": node.get("title"),
                    "caption": cap_txt,
                    "table_schema": schema,
                    "aliases": aliases_text,
                    "labels": node.get("label"),
                    "body": node.get("dense_text"),
                    "filters": {
                        **({"page_idx": node.get("page_idx")} if node.get("page_idx") is not None else {}),
                        **({"parent_section": node.get("parent_section")} if node.get("parent_section") else {}),
                        **({"units_set": units_set} if units_set else {}),
                    },
                })
                if role == "image":
                    spans = _build_figure_spans(
                        orig,
                        parent_section=node.get("parent_section"),
                        parent_title=node.get("parent_title"),
                    )
                    if spans:
                        figure_spans.extend(spans)
                        for span in spans:
                            graph_edges.append({
                                "src": nid,
                                "dst": span["span_id"],
                                "type": "has_span",
                                "span_role": span.get("role"),
                            })
                elif role == "table":
                    headers, row_values = _extract_table_struct(orig)
                    row_records = _build_table_row_records(
                        orig,
                        headers,
                        row_values,
                        parent_section=node.get("parent_section"),
                        parent_title=node.get("parent_title"),
                    )
                    if row_records:
                        table_cells.extend(row_records)
                        for rec in row_records:
                            graph_edges.append({
                                "src": nid,
                                "dst": rec["row_id"],
                                "type": "has_row",
                            })
            # Graph edges: already appended for image/table via parent_section
        else:
            # Skip adding non-chunk leaf entries to dense/sparse views to avoid duplication.
            # Leaf chunks will be added in the projection stage below.
            pass
            
    # Also project leaf chunk entries into dense/sparse + ref edges
    for e in leaf_entries:
        nid = e.get("node_id")
        role = e.get("role")
        raw = e.get("raw_text")
        page_idx = e.get("page_idx")
        parent_section = e.get("parent_section")
        # Resolve parent title for better context in MMR
        parent_title_ctx = None
        if parent_section and parent_section in id2node:
            pt = id2node[parent_section].get("title")
            if isinstance(pt, str) and pt.strip():
                parent_title_ctx = pt.strip()
        # refs via label detection
        refs = None
        lkind, lbl, num = _extract_label(raw or "")
        if lbl:
            target = id_maps["label2id"].get(lbl)
            if not target and lkind == "figure" and num:
                target = id_maps["figure"].get(str(num))
            if not target and lkind == "table" and num:
                target = id_maps["table"].get(str(num))
            if target:
                refs = [{"type": "ref", "label": lbl, "target": target}]
                graph_edges.append({"src": nid, "dst": target, "type": "ref", "label": lbl})

        units_set = _extract_units_set(raw or "")
        # Dense text for chunk via MMR (no粗暴截断)
        dense_text_chunk = summarize_text_block(
            title=parent_title_ctx or "",
            raw=(raw or "")[:800],
            budget_chars=220,
            k=2,
            keywords=_hints_from_text(raw or "", k=6),
        )
        dense_leaf.append({
            "node_id": nid,
            "orig_node_id": e.get("orig_node_id") or nid,
            "role": role,
            "dense_text": dense_text_chunk or (raw or ""),
            "raw_text": raw or None,
            "filters": {
                **({"page_idx": page_idx} if page_idx is not None else {}),
                **({"parent_section": parent_section} if parent_section else {}),
                **({"units_set": units_set} if units_set else {}),
            },
            "refs": refs,
        })
        sparse_leaf.append({
            "id": nid,
            "role": role,
            "title": None,
            "caption": None,
            "table_schema": None,
            "aliases": _build_aliases_text(raw or ""),
            "labels": e.get("label"),
            "body": raw or None,
            "filters": {
                **({"page_idx": page_idx} if page_idx is not None else {}),
                **({"parent_section": parent_section} if parent_section else {}),
                **({"units_set": units_set} if units_set else {}),
            },
        })
        if isinstance(page_idx, int):
            graph_edges.append({"src": f"P{page_idx}", "dst": nid, "type": "same_page"})

    # prev_next edges by page using read_order_idx
    page_groups: Dict[int, List[Tuple[int, str]]] = {}
    for _nid, _node in id2node.items():
        p = _node.get("page_idx")
        o = _node.get("read_order_idx")
        if isinstance(p, int) and isinstance(o, int):
            page_groups.setdefault(p, []).append((o, _nid))
    for p, arr in page_groups.items():
        arr.sort(key=lambda x: x[0])
        for i in range(len(arr) - 1):
            a = arr[i][1]; b = arr[i+1][1]
            graph_edges.append({"src": a, "dst": b, "type": "prev_next"})

    keyword_map_clean = {k: _unique(v) for k, v in heading_keyword_map.items()}
    heading_index = {
        "heading_titles": heading_titles,
        "heading_paths": heading_paths_map,
        "heading_children": heading_children,
        "keyword_map": keyword_map_clean,
    }
    meta_conf = summary_obj.get("meta", {})
    metadata = {
        "doc_id": doc_id,
        "generated_at": meta_conf.get("ts"),
        "summary_model": meta_conf.get("model"),
        "include_leaves": bool(include_leaves),
        "use_llm": bool(use_llm),
        "llm_model": meta_conf.get("llm_model"),
        "counts": {
            "dense_coarse": len(dense_coarse),
            "dense_leaf": len(dense_leaf),
            "sparse_coarse": len(sparse_coarse),
            "sparse_leaf": len(sparse_leaf),
            "table_cells": len(table_cells),
            "figure_spans": len(figure_spans),
            "graph_edges": len(graph_edges),
        },
    }

    return (
        summary_obj,
        coarse_entries,
        leaf_entries,
        dense_coarse,
        dense_leaf,
        sparse_coarse,
        sparse_leaf,
        table_cells,
        figure_spans,
        graph_edges,
        heading_index,
        metadata,
        {"label2id": id_maps["label2id"], "figure": id_maps["figure"], "table": id_maps["table"]},
    )

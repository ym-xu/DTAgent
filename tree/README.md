MM DocTree Adapter (MinerU content_list)

Overview
- Purpose: Normalize and enrich MinerU content_list for later MM DocTree building.
- Scope now: Deterministic adapter only; no TOC/heading/caption detection yet.
- Key additions: node_idx, node_level (rename), outline/bbox for images/tables, optional OCR/LLM hooks. Also extracts Markdown preview for tables.
  Also extracts Markdown preview for tables.

Recent Additions (LLM, Slides)
- Page images rendering: builder renders all PDF pages to `<source>/images/page_{idx}.png` with a progress bar.
- Mode detection (doc vs slides): GPT‑4o by default. Samples page 0 plus one random page. Results on root:
  - `mode` ∈ {"doc","slides"}
  - `mode_confidence` ∈ [0,1]
  - `columns`: 1 or 2 where 2 means a two‑up scan (two logical pages on one physical page).
- Slides tree:
  - `build_slides_tree(flat_root)`: per‑page image node as slide container with page’s elements under it.
  - `build_tree_by_mode(flat_root, mode)`: returns slides tree when mode="slides", else page tree.
- TOC LLM payload cleanup: when asking LLM to judge TOC pages, we no longer include x/y geometry; only text is sent.


What It Does
- Adds `node_idx` sequentially (0..n-1).
- Renames `text_level` to `node_level` when present; keeps other fields intact.
- Sets `node_level = -1` for any item missing it (including non-title `text` and non-text modalities).
- Enriches `image`/`table` items with `outline` (bbox) from `layout.json` by image filename matching.
- Provides optional hooks for OCR and LLM description (not enabled by default).
- Ensures `image` items always contain `text` and `description` fields (empty if OCR/LLM not used).
- Extracts `table_text` from `table_body` HTML as Markdown when possible; falls back to plain text.
- Sanitizes text fields to remove zero-width/control characters, normalize whitespace, and drop long runs of single-letter noise.
- Ensures caption/footnote lists exist for images/tables (`image_caption`, `image_footnote`, `table_caption`, `table_footnote`).
  Entries are sanitized and deduplicated; empty/garbled items are removed.

Input Example (MinerU content_list)
```
[
  {
    "type": "text",
    "text": "UndergraduateProspectus",
    "text_level": 1,
    "page_idx": 0
  },
  {
    "type": "image",
    "img_path": "images/ca5dfe45....jpg",
    "image_caption": [],
    "image_footnote": [],
    "page_idx": 1
  },
  {
    "type": "table",
    "img_path": "images/f0f54084....jpg",
    "table_caption": ["APPLICATION PERIOD AND FEE"],
    "table_footnote": [],
    "table_body": "<table>...</table>",
    "page_idx": 24
  },
  {
    "type": "equation",
    "text": "\n$$\nI_{l} = \\left|\\sum_{i}A_{h,i}\\odot \\frac{\\partial\\mathcal{L}(x)}{\\partial A_{h,i}}\\right|. \\tag{1}\n$$\n",
    "text_format": "latex",
    "page_idx": 1
  }
]
```

Output Example (adapted)
```
[
  {
    "type": "text",
    "text": "UndergraduateProspectus",
    "node_level": 1,
    "page_idx": 0,
    "node_idx": 0
  },
  {
    "type": "image",
    "img_path": "images/ca5dfe45....jpg",
    "image_caption": [],
    "image_footnote": [],
    "page_idx": 1,
    "node_level": -1,
    "node_idx": 1,
    "outline": [x0, y0, x1, y1],
    "text": "",                         // may contain OCR text
    "description": ""                  // may contain LLM description
  },
  {
    "type": "table",
    "img_path": "images/f0f54084....jpg",
    "table_caption": ["APPLICATION PERIOD AND FEE"],
    "table_footnote": [],
    "table_body": "<table>...</table>",
    "page_idx": 24,
    "node_level": -1,
    "node_idx": 2,
    "outline": [x0, y0, x1, y1],
    "table_text": "| Col1 | Col2 |\n| --- | --- |\n| v1 | v2 |"  
  },
  {
    "type": "equation",
    "text": "\n$$\nI_{l} = \\left|\\sum_{i}A_{h,i}\\odot \\frac{\\partial\\mathcal{L}(x)}{\\partial A_{h,i}}\\right|. \\tag{1}\n$$\n",
    "text_format": "latex",
    "page_idx": 1,
    "node_level": -1,
    "node_idx": 3
  }
]
```

CLI Usage
- Directory batch:
  - Default (keep originals): `python -m tree.adapter --in-dir /path/to/MinerU_MMLB`
    - Writes alongside as `content_list.adapted.json`
    - Customize name: `python -m tree.adapter --in-dir /path/to/MinerU_MMLB --suffix ".mm.json"` → `content_list.mm.json`
  - Overwrite originals: `python -m tree.adapter --in-dir /path/to/MinerU_MMLB --in-place`
- Single file:
  - Default (keep original): `python -m tree.adapter --in-file /dir/content_list.json --layout-file /dir/layout.json`
    - Writes `/dir/content_list.adapted.json` unless you pass `--out` or customize `--suffix`
  - Overwrite original: `python -m tree.adapter --in-file /dir/content_list.json --layout-file /dir/layout.json --in-place`

Python API
```
from tree.adapter import adapt_content_list, adapt_single_file
from tree.utils import try_ocr_image, describe_image_with_llm

# Adapt an in-memory list
adapted = adapt_content_list(
    content_list,
    layout_obj=layout,                     # dict loaded from layout.json
    ocr_image_func=None,                   # or try_ocr_image
    describe_image_func=None               # or your LLM hook
)

# Adapt one file (write back)
adapt_single_file(
    dom_file="/dir/content_list.json",
    layout_file="/dir/layout.json",
    ocr_image_func=None,
    describe_image_func=None,
    write_in_place=True,
)
```

Flat DocTree Builder
- Purpose: Turn adapted `content_list` into a flat `doctree.json` per PDF directory for downstream retrieval/visualization.
- CLI:
  - Directory mode: `python -m tree.builder --in-dir /path/to/MinerU_MMLB`
    - Writes `/dir/of/pdf/doctree.json` for each document directory.
    - Customize output name: `--suffix ".flat.json"` → `doctree.flat.json`.
    - Prefer adapted inputs via `--adapted-suffix ".adapted.json"` (default).
  - Single file: `python -m tree.builder --in-file /dir/content_list.adapted.json --out /dir/doctree.json`
  - Optional flags:
    - `--consolidate-toc` Detect and consolidate TOC into a single toc node (skipped when slides)

Usage Quickstart
- Typical flow (per document directory):
  1) Adapt MinerU outputs (optional but recommended for outlines/OCR):
     - `python -m tree.adapter --in-dir /path/to/MinerU_MMLB`
  2) Build flat doctree (renders page images, detects mode with GPT‑4o):
     - `python -m tree.builder --in-dir /path/to/MinerU_MMLB --suffix ".flat.json"`
  3) Enable TOC consolidation when needed:
     - `python -m tree.builder --in-dir /path/to/MinerU_MMLB --suffix ".flat.json" --consolidate-toc`
  4) Single file mode:
     - `python -m tree.builder --in-file /dir/content_list.adapted.json --out /dir/doctree.json --consolidate-toc`

- Programmatic API
  - Build doctree with options:
    - `from tree.builder import build_flat_doctree`
    - `tree = build_flat_doctree(adapted_list, source_dir=doc_dir, doc_id=doc_id, consolidate_toc=True)`
  - Slides tree view:
    - `from tree.trees import build_slides_tree, build_tree_by_mode`
    - `slides = build_slides_tree(tree)` or `view = build_tree_by_mode(tree, mode=tree.get("mode","doc"))`
  - TOC helpers (already wired in builder):
    - `from tree.toc import find_toc_pages, consolidate_toc_v2`
    - `pages = find_toc_pages(tree, llm_call=gpt_llm_call)` then `tree = consolidate_toc_v2(tree, pages, gpt_llm_call)`

- Output fields (builder):
  - Root:
    - `mode`: "doc" or "slides" (by GPT‑4o)
    - `mode_confidence`: float in [0,1]
    - `columns`: 1 or 2 (2 means a two‑up scan detected)
    - `children`: flat nodes; with `--consolidate-toc`, TOC lines may be replaced by a single `toc` node (`headings`, `pages`)
  - Node: original MinerU nodes with added indexing fields
- DocTree root schema (minimal):
  - `type: "document"`
  - `doc_id`: directory name of the PDF folder
  - `source_path`: directory path
  - `children`: flat list of nodes in reading order
  - `indices` (enabled by default):
    - `by_page: {page_idx -> [node_idx...]}`
    - `by_type: {type -> [node_idx...]}`
    - `id_to_idx: {"{doc_id}#{node_idx}" -> node_idx}`
  - Note: JSON keys are strings; `by_page` keys originate as ints and will be serialized as JSON object keys.

Children schema (selected fields):
- Common: `type`, `node_id`, `node_idx`, `page_idx`, `node_level`
- Text: `text`
- Image: `img_path`, `outline?`, `text`, `description`
- Table: `img_path?`, `outline?`, `table_body`, `table_text?`
- Equation: `text`, `text_format?`

Page Refinement APIs (LLM-assisted, no CLI)
- Goal: Apply page-level refinements (reorder, relevel, optional merge/add title) via LLM outputs while keeping the final doctree flat schema unchanged.
- Two paths are supported:
  - Plan path (recommended): model outputs a strict plan object `{page_idx, items, merges?, virtual_titles?}`; we validate and apply it.
  - Jsonlist path (simple): model outputs an array `[{node_id, level}]` for reorder + relevel only.
- Example (plan path):
  ```python
  import json
  from tree import (
      build_page_payload,
      render_plan_prompt,
      validate_and_normalize_plan,
      apply_plan_to_document,
  )

  root = json.load(open('/path/to/doctree.json','r',encoding='utf-8'))
  page_idx = 11
  # Option A: no image (simple)
  payload = build_page_payload(root, page_idx)

  # Option B: try to find an existing page image near source_path (common names like page_0.png, page_1.png)
  # We search under <source_path>/images first
  payload = build_page_payload(root, page_idx, page_image_mode="find")

  # Build Page/Chapter Trees
  from tree import build_page_tree, build_chapter_tree
  page_tree = build_page_tree(flat)
  chapter_tree = build_chapter_tree(flat, max_level=6)

Layout → outline extraction
- Searches `layout.json` (or `layout['pdf_info']` when present) recursively for entries with `image_path` and sibling `bbox`.
- Builds an index and matches by image filename (basename). If exact match fails, uses partial path inclusion as fallback.
- Adds `outline` to the content_list item when a match is found.

OCR / LLM Hooks (Optional)
- OCR: `tree.utils.try_ocr_image(path)` uses Pillow + pytesseract if available; returns text or None.
  - Dependencies: `pip install pillow pytesseract` and system tesseract installed.
- LLM (VLM) Description: `tree.utils.describe_image_with_llm(path, prompt=None, **kwargs)` is a placeholder; implement with your client (e.g., qwen2.5-vl) and return a short description.
- Hooks are not called by default; pass them explicitly in API calls.

Notes
- `page_idx` is preserved as-is from MinerU.
- The adapter does not perform title disambiguation, TOC detection, caption pairing, or list nesting inference.
- Unknown fields from MinerU are preserved untouched.
 - Builder always renders all pages to images and shows a progress bar; GPT‑4o mode detection runs by default.
 - For LLM features set `OPENAI_API_KEY` in the environment.

LLM & Dependencies
- Mode detection and logical page labeling use GPT‑4o; set `OPENAI_API_KEY` in the environment.
- Qwen‑based paths were removed; GPT‑4o is the only LLM used here.

Planned Next (Enhance Stage)
- Heading/TOC detection, list nesting with stack, caption pairing (figure/table), equations and code blocks refinement.
- LLM-assisted heuristics for ambiguous cases.

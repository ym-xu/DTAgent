"""
Tree package: adapter to enrich MinerU content_list for MM DocTree.

Focus of the adapter step:
- Add node_idx (sequential index)
- Rename MinerU's text_level -> node_level when present
- Default node_level = -1 for non-text modalities (image/table/equation/code)
- Attach outline/bbox for images/tables by reading layout.json
- Optional interfaces for OCR and LLM description are provided in utils.py
"""

from .adapter import (
    adapt_content_list,
    adapt_single_file,
    enrich_content_with_layout,
    process_directory,
)
from .builder import (
    build_flat_doctree,
    build_single as build_doctree_single,
    build_directory as build_doctree_directory,
)
from .trees import (
    build_page_tree,
    build_chapter_tree,
)
from .toc import (
    build_toc_page_payload,
    render_toc_detect_prompt,
    detect_toc_page,
    find_toc_pages,
    render_toc_parse_prompt,
    build_toc_tree_with_llm,
)
from .llm_clients import (
    gpt_llm_call,
    qwen_llm_call,
)

__all__ = [
    "adapt_content_list",
    "adapt_single_file",
    "enrich_content_with_layout",
    "process_directory",
    "build_flat_doctree",
    "build_doctree_single",
    "build_doctree_directory",
    "build_page_tree",
    "build_chapter_tree",
    "build_toc_page_payload",
    "render_toc_detect_prompt",
    "detect_toc_page",
    "find_toc_pages",
    "render_toc_parse_prompt",
    "build_toc_tree_with_llm",
    "gpt_llm_call",
    "qwen_llm_call",
]

__version__ = "0.5.0"

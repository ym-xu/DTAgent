"""
Tree package: adapter to enrich MinerU content_list for MM DocTree.

Focus of the adapter step:
- Add node_idx (sequential index)
- Rename MinerU's text_level -> node_level when present
- Default node_level = -1 for non-text modalities (image/table/equation/code)
- Attach outline/bbox for images/tables by reading layout.json
- Optional interfaces for OCR and LLM description are provided in utils.py
"""

# from .adapter import (
#     adapt_content_list,
#     adapt_single_file,
#     enrich_content_with_layout,
#     process_directory,
# )
from .adapter_v2 import (
    adapt_content_list_v2,
    adapt_single_file_v2,
    process_directory_v2,
)
# from .trees import (
#     build_page_tree,
#     build_chapter_tree,
#     build_slides_tree,
#     build_tree_by_mode,
# )
from .headings import (
    merge_adjacent_headings,
)
from .toc import (
    build_toc_page_payload,
    render_toc_detect_prompt,
    detect_toc_page,
    find_toc_pages,
    build_toc_pages_payload_with_nodes,
    render_toc_parse_with_span_prompt,
    build_toc_tree_and_span_with_llm,
    consolidate_toc_v2,
)
from .llm_clients import (
    gpt_llm_call,
    qwen_llm_call,
)

from .utils_2 import (
    print_heading_outline
)

__all__ = [
    "adapt_content_list",
    "adapt_single_file",
    "enrich_content_with_layout",
    "process_directory",
    "adapt_content_list_v2",
    "adapt_single_file_v2",
    "process_directory_v2",
    "build_flat_doctree",
    "build_doctree_single",
    "build_doctree_directory",
    "build_page_tree",
    "build_chapter_tree",
    "build_slides_tree",
    "build_tree_by_mode",
    "merge_adjacent_headings",
    "build_toc_page_payload",
    "render_toc_detect_prompt",
    "detect_toc_page",
    "find_toc_pages",
    "build_toc_pages_payload_with_nodes",
    "render_toc_parse_with_span_prompt",
    "build_toc_tree_and_span_with_llm",
    "consolidate_toc_v2",
    "gpt_llm_call",
    "qwen_llm_call",
    "print_heading_outline",
]

__version__ = "0.5.0"

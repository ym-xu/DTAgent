import os
import json
from tqdm import tqdm


class BlockBuilder:
    """Encapsulates logic for building retrieval blocks from DOM files."""

    def __init__(self, dataset):
        """
        Args:
            dataset: BaseDataset instance (provides config, time, and utilities)
        """
        self.dataset = dataset

    def build_block_tree(self, block_config):
        """
        构建block tree：将DOM elements按章节层级组织成适合检索的blocks

        Args:
            block_config: block tree配置
        """
        input_dir = block_config.input_path
        output_dir = block_config.output_path

        if not os.path.exists(input_dir):
            print(f"Input directory not found: {input_dir}")
            return

        dom_files = [f for f in os.listdir(input_dir) if f.endswith('_with_descriptions.json')]
        print(f"Found {len(dom_files)} DOM files to process")

        os.makedirs(output_dir, exist_ok=True)

        for dom_file in tqdm(dom_files, desc="Building block trees"):
            try:
                dom_file_path = os.path.join(input_dir, dom_file)
                base_name = dom_file.replace('_with_descriptions.json', '')
                output_path = os.path.join(output_dir, f"{base_name}_blocks.json")

                self._process_single_dom_to_blocks(dom_file_path, output_path, block_config)
            except Exception as e:
                print(f"Error processing DOM file {dom_file}: {e}")
                continue

    def _process_single_dom_to_blocks(self, dom_file_path, output_path, block_config):
        """
        处理单个DOM文件，构建blocks

        Args:
            dom_file_path: 输入的_with_descriptions.json文件路径
            output_path: 输出blocks文件路径
            block_config: 配置
        """
        dom_data = self.dataset.load_dom_nodes(dom_file_path)

        # 获取elements
        elements = dom_data.get('data', {}).get('elements', [])
        if not elements:
            print(f"No elements found in {dom_file_path}")
            return

        # 分析章节结构
        chapter_hierarchy = self._analyze_chapter_hierarchy(elements)
        leaf_chapters = self._find_leaf_chapters(chapter_hierarchy)

        # 构建blocks
        blocks = self._build_chapter_blocks(elements, leaf_chapters, chapter_hierarchy, block_config)

        # 处理孤立元素
        orphan_blocks = self._handle_orphan_elements(elements, block_config)
        blocks.extend(orphan_blocks)

        # 保存结果
        base_name = os.path.splitext(os.path.basename(dom_file_path))[0].replace('_with_descriptions', '')
        doc_name = base_name
        result = {
            "metadata": {
                "doc_name": doc_name,
                "total_blocks": len(blocks),
                "total_elements": len(elements),
                "build_config": {
                    "max_words": block_config.max_words,
                    "min_words": block_config.min_words,
                    "include_parent_context": block_config.include_parent_context
                },
                "timestamp": self.dataset.time
            },
            "blocks": blocks
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"Created {len(blocks)} blocks -> {output_path}")

    def _analyze_chapter_hierarchy(self, elements):
        """
        分析章节层级结构

        Returns:
            dict: {chapter_id: {"title": str, "parent": int, "children": list}}
        """
        hierarchy = {}

        # 收集所有章节标题
        for element in elements:
            if element.get('is_chapter_title'):
                chapter_id = element['index']
                hierarchy[chapter_id] = {
                    "title": element['text'],
                    "parent": element['parent_chapter'],
                    "children": [],
                    "element": element
                }

        # 建立父子关系
        for chapter_id, info in hierarchy.items():
            parent_id = info['parent']
            if parent_id != -1 and parent_id in hierarchy:
                hierarchy[parent_id]['children'].append(chapter_id)

        return hierarchy

    def _find_leaf_chapters(self, chapter_hierarchy):
        """找到叶子章节（没有子章节的最底层章节）"""
        leaf_chapters = []
        for chapter_id, info in chapter_hierarchy.items():
            if not info['children']:
                leaf_chapters.append(chapter_id)
        return leaf_chapters

    def _build_chapter_blocks(self, elements, leaf_chapters, chapter_hierarchy, block_config):
        """
        为所有章节构建blocks（修复版本：不再只处理叶子章节）
        """
        blocks = []
        all_chapters = list(chapter_hierarchy.keys())
        for chapter_id in all_chapters:
            chapter_elements = self._get_chapter_elements(elements, chapter_id)
            if not chapter_elements:
                continue
            word_count = self._calculate_word_count(chapter_elements)
            if word_count <= block_config.max_words:
                block = self._create_block(
                    f"chapter_{chapter_id}",
                    chapter_elements,
                    chapter_hierarchy,
                    chapter_id
                )
                blocks.append(block)
            else:
                sub_blocks = self._split_large_chapter(
                    chapter_id,
                    chapter_elements,
                    chapter_hierarchy,
                    block_config
                )
                blocks.extend(sub_blocks)
        return blocks

    def _get_chapter_elements(self, elements, chapter_id):
        """获取属于指定章节的直接内容元素（避免重复收集子章节内容）"""
        child_chapter_ids = set()
        for element in elements:
            if element.get('is_chapter_title') and element.get('parent_chapter') == chapter_id:
                child_chapter_ids.add(element['index'])

        chapter_elements = []
        for element in elements:
            if element.get('is_chapter_title') and element['index'] == chapter_id:
                chapter_elements.append(element)
            elif (element.get('parent_chapter') == chapter_id and 
                  element['index'] not in child_chapter_ids):
                chapter_elements.append(element)

        chapter_elements.sort(key=lambda x: x['index'])
        return chapter_elements

    def _calculate_word_count(self, elements):
        """计算元素列表的总词数（使用原始词数统计）"""
        total_words = 0
        for element in elements:
            text = element.get('text', '') + ' ' + element.get('description', '')
            raw_words = len(text.split())
            total_words += raw_words
        return int(total_words)

    def _create_block(self, block_id, elements, chapter_hierarchy, chapter_id):
        """创建单个block"""
        parent_path = self._build_parent_path(chapter_hierarchy, chapter_id)
        text_count = len([e for e in elements if e.get('type') == 'paragraph'])
        # Count both 'figure' and MinerU 'image'
        figure_count = len([e for e in elements if e.get('type') in ('figure', 'image')])
        table_count = len([e for e in elements if e.get('type') == 'table'])

        pages = [e.get('page', 0) for e in elements]
        page_range = [min(pages), max(pages)] if pages else [0, 0]

        return {
            "block_id": block_id,
            "block_type": "chapter_section",
            "title": chapter_hierarchy[chapter_id]['title'],
            "parent_path": parent_path,
            "word_count": self._calculate_word_count(elements),
            "element_count": len(elements),
            "content_stats": {
                "text_count": text_count,
                "figure_count": figure_count,
                "table_count": table_count
            },
            "page_range": page_range,
            "elements": elements
        }

    def _build_parent_path(self, chapter_hierarchy, chapter_id):
        """构建父章节路径"""
        path = []
        current_id = chapter_id
        while current_id != -1 and current_id in chapter_hierarchy:
            path.insert(0, chapter_hierarchy[current_id]['title'])
            current_id = chapter_hierarchy[current_id]['parent']
        return path

    def _split_large_chapter(self, chapter_id, elements, chapter_hierarchy, block_config):
        """分割超长章节"""
        title_element = None
        content_elements = []
        for element in elements:
            if element.get('is_chapter_title') and element['index'] == chapter_id:
                title_element = element
            else:
                content_elements.append(element)
        if not title_element:
            return []

        blocks = []
        current_elements = [title_element] if block_config.get('include_parent_context', True) else []
        current_words = self._calculate_word_count(current_elements)
        part_num = 1

        for element in content_elements:
            element_words = self._calculate_word_count([element])
            if current_words + element_words <= block_config.max_words:
                current_elements.append(element)
                current_words += element_words
            else:
                block = self._create_block(
                    f"chapter_{chapter_id}_part_{part_num}",
                    current_elements,
                    chapter_hierarchy,
                    chapter_id
                )
                blocks.append(block)

                part_num += 1
                current_elements = [title_element] if block_config.get('include_parent_context', True) else []
                current_words = self._calculate_word_count(current_elements)

                current_elements.append(element)
                current_words += element_words

        if len(current_elements) > (1 if title_element else 0):
            block = self._create_block(
                f"chapter_{chapter_id}_part_{part_num}",
                current_elements,
                chapter_hierarchy,
                chapter_id
            )
            blocks.append(block)

        return blocks

    def _handle_orphan_elements(self, elements, block_config):
        """处理孤立元素（parent_chapter=-1且非标题）"""
        orphan_elements = [
            e for e in elements 
            if e.get('parent_chapter') == -1 and not e.get('is_chapter_title')
        ]
        if not orphan_elements:
            return []

        page_groups = {}
        for element in orphan_elements:
            page = element.get('page', 0)
            if page not in page_groups:
                page_groups[page] = []
            page_groups[page].append(element)

        blocks = []
        for page, page_elements in page_groups.items():
            block = {
                "block_id": f"orphan_page_{page}",
                "block_type": "orphan_content",
                "title": f"Page {page} Content",
                "parent_path": [],
                "word_count": self._calculate_word_count(page_elements),
                "element_count": len(page_elements),
                "content_stats": {
                    "text_count": len([e for e in page_elements if e.get('type') == 'paragraph']),
                    "figure_count": len([e for e in page_elements if e.get('type') in ('figure', 'image')]),
                    "table_count": len([e for e in page_elements if e.get('type') == 'table'])
                },
                "page_range": [page, page],
                "elements": page_elements
            }
            blocks.append(block)

        return blocks

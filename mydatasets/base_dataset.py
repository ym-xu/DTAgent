import json
import re   
from dataclasses import dataclass
import os
from tqdm import tqdm
import pymupdf
import glob
from PIL import Image
from datetime import datetime


@dataclass
class Content:
    image: Image
    image_path: str
    txt: str

class BaseDataset():
    def __init__(self, config):
        self.config = config
        self.IMG_FILE = lambda doc_name,index: f"{self.config.extract_path}/page_image/{doc_name}_{index}.png"
        self.TEXT_FILE = lambda doc_name,index: f"{self.config.extract_path}/page_text/{doc_name}_{index}.txt"
        self.EXTRACT_DOCUMENT_ID = lambda sample: re.sub("\\.pdf$", "", sample["doc_id"]).split("/")[-1] 
        current_time = datetime.now()
        self.time = current_time.strftime("%Y-%m-%d-%H-%M")

    def enrich_content_with_layout(self, dom_files, layout_files):
        """使用layout信息丰富content_list，为图像添加bbox信息"""
        # 创建layout文件的字典，以目录为键
        layout_dict = {}
        for layout_file in layout_files:
            layout_dir = os.path.dirname(layout_file)
            layout_dict[layout_dir] = layout_file
        
        print(f"Found {len(layout_dict)} layout directories")
        
        # 处理每个content_list文件
        success_count = 0
        error_count = 0
        
        for dom_file in tqdm(dom_files, desc="Enriching content lists with layout info"):
            try:
                dom_dir_path = os.path.dirname(dom_file)
                
                # 找到对应的layout文件
                if dom_dir_path in layout_dict:
                    layout_file = layout_dict[dom_dir_path]
                    
                    # 加载layout数据
                    with open(layout_file, 'r', encoding='utf-8') as f:
                        layout = json.load(f)
                    
                    # 加载content_list数据
                    with open(dom_file, 'r', encoding='utf-8') as f:
                        content_list = json.load(f)
                    
                    # 提取layout信息
                    layout_info = {}
                    def recursive_search(obj):
                        if isinstance(obj, dict):
                            for key, value in obj.items():
                                if key == "image_path" and isinstance(value, str):
                                    if "bbox" in obj:
                                        layout_info[value] = obj["bbox"]
                                else:
                                    recursive_search(value)
                        elif isinstance(obj, list):
                            for item in obj:
                                recursive_search(item)
                    
                    recursive_search(layout.get('pdf_info', layout))
                    
                    # 丰富content_list
                    enriched_count = 0
                    for item in content_list:
                        if "img_path" in item:
                            img_filename = item['img_path'].split('/')[-1]
                            if img_filename in layout_info:
                                item["outline"] = layout_info[img_filename]
                                enriched_count += 1
                            else:
                                # 尝试部分匹配
                                for layout_path, bbox in layout_info.items():
                                    if img_filename in layout_path:
                                        item["outline"] = bbox
                                        enriched_count += 1
                                        break
                    
                    # 保存丰富后的content_list
                    with open(dom_file, 'w', encoding='utf-8') as f:
                        json.dump(content_list, f, ensure_ascii=False, indent=2)
                    
                    if enriched_count > 0:
                        print(f"Enriched {enriched_count} items in {os.path.basename(dom_file)}")
                    success_count += 1
                    
                else:
                    print(f"No layout file found for {dom_file}")
                    error_count += 1
                    
            except Exception as e:
                print(f"Error processing {dom_file}: {e}")
                error_count += 1
                continue
        
        print(f"\nEnrichment completed: {success_count} success, {error_count} errors")

    def load_data(self, use_retreival=True):
        path = self.config.sample_path
        if use_retreival:
            try:
                assert(os.path.exists(self.config.sample_with_retrieval_path))
                path = self.config.sample_with_retrieval_path
            except:
                print("Use original sample path!")
                
        assert(os.path.exists(path))
        with open(path, 'r') as f:
            samples = json.load(f)
            
        return samples

    # ToDo: read pdf, recover to dict/html, extract dom nodes, save to DOM_FILE
    # build tree (block, hybrid): input pdf, output dict/html (tem/dataname/tree_method/doc_id.json/html and img_dir)
    #   - change MdocAgent extraction: tem/dataname/{add page_text}/page_text.txt and tem/dataname/{add page_image}/page_text.txt
    # extract_nodes: input dict/html, output json, save to DOM_FILE (tem/dataname/node_method/doc_id.json)
        
    # MDocAgent text and image extract
    def extract_content(self, resolution=144):
        samples = self.load_data()
        for sample in tqdm(samples):
            self._extract_content(sample, resolution=resolution)

    def _extract_content(self, sample, resolution=144):
        max_pages=self.config.max_page
        image_list = list()
        text_list = list()
        doc_name = self.EXTRACT_DOCUMENT_ID(sample)
        os.makedirs(self.config.extract_path, exist_ok=True)
        with pymupdf.open(os.path.join(self.config.document_path, sample["doc_id"])) as pdf:
            for index, page in enumerate(pdf[:max_pages]):
                img_file = self.IMG_FILE(doc_name,index)
                if not os.path.exists(img_file):
                    os.makedirs(os.path.dirname(img_file), exist_ok=True)
                    im = page.get_pixmap(dpi=resolution)
                    im.save(img_file)
                image_list.append(img_file)
                txt_file = self.TEXT_FILE(doc_name,index)
                if not os.path.exists(txt_file):
                    os.makedirs(os.path.dirname(txt_file), exist_ok=True)
                    text = page.get_text("text")
                    with open(txt_file, 'w') as f:
                        f.write(text)
                text_list.append(txt_file)
        return image_list, text_list

    def load_processed_content(self, sample: dict, disable_load_image=True)->list[Content]:
        # ToDo: check dom mode, download dom node json file, return text
        doc_name = self.EXTRACT_DOCUMENT_ID(sample)
        content_list = []
        for page_idx in range(self.config.max_page):
            img_file = self.IMG_FILE(doc_name, page_idx)
            text_file = self.TEXT_FILE(doc_name, page_idx)
            if not os.path.exists(img_file):
                break
            img = None
            if not disable_load_image:
                img = self.load_image(img_file)
            txt = self.load_txt(text_file)
            content_list.append(Content(image=img, image_path=img_file, txt=txt)) 
        return content_list

    def load_image(self, file):
        pil_im = Image.open(file)
        return pil_im

    def load_txt(self, file):
        max_length = self.config.max_character_per_page
        with open(file, 'r') as file:
            content = file.read()
        content = content.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
        return content[:max_length]

    # read json, return [node1, node2, ...], consider meta data
    def load_dom_nodes(self, file):
        """
        从DOM JSON文件中加载节点数据
        
        Args:
            file: DOM JSON文件路径
            
        Returns:
            DOM数据字典
        """
        if not os.path.exists(file):
            raise FileNotFoundError(f"DOM file not found: {file}")
        
        with open(file, 'r', encoding='utf-8') as f:
            dom_data = json.load(f)
        
        return dom_data

    def get_dom_file_path(self, sample: dict) -> str:
        """
        获取样本对应的DOM文件路径
        
        Args:
            sample: 数据样本字典
            
        Returns:
            DOM文件路径
        """
        if hasattr(self.config, 'dom_path'):
            doc_name = self.EXTRACT_DOCUMENT_ID(sample)
            return os.path.join(self.config.dom_path, f"{doc_name}.json")
        else:
            raise AttributeError("dom_path not configured in dataset config")

    def process_dom_images(self, dom_agent, dom_config):
        """
        处理DOM文件中的图像元素，为其添加描述
        
        Args:
            dom_agent: DomImageAgent实例
            dom_config: DOM处理相关配置
        """
        if dom_config.dom_method == "chatdoc":
            dom_dir = self.config.dom_path
            if not os.path.exists(dom_dir):
                print(f"DOM directory not found: {dom_dir}")
                return
            
            dom_files = [f for f in os.listdir(dom_dir) if f.endswith('.json')]
            print(f"Found {len(dom_files)} DOM files in {dom_dir}")
            
            for dom_file in tqdm(dom_files, desc="Processing DOM files"):
                try:
                    dom_file_path = os.path.join(dom_dir, dom_file)
                    pdf_name = dom_file.replace('.json', '.pdf')
                    pdf_file_path = os.path.join(self.config.document_path, pdf_name)
                    
                    if not os.path.exists(pdf_file_path):
                        print(f"PDF file not found: {pdf_file_path}")
                        continue
                    
                    self._process_single_dom_file(dom_agent, dom_file_path, pdf_file_path, dom_config)
                    
                except Exception as e:
                    print(f"Error processing DOM file {dom_file}: {e}")
                    continue

        elif dom_config.dom_method == "mineru":
            dom_dir = self.config.dom_path
            if not os.path.exists(dom_dir):
                print(f"DOM directory not found: {dom_dir}")
                return

            dom_files = glob.glob(os.path.join(dom_dir, '**/*content_list.json'), recursive=True)
            print(f"Found {len(dom_files)} DOM files in {dom_dir}")

            layout_files = glob.glob(os.path.join(dom_dir, '**/*layout.json'), recursive=True)

            self.enrich_content_with_layout(dom_files, layout_files)

            for dom_file in tqdm(dom_files, desc="Processing DOM files"):
                try:
                    dom_file_path = os.path.join(dom_dir, dom_file)
                    pdf_name = dom_file.split('/')[-2]
                    pdf_file_path = os.path.join(self.config.document_path, pdf_name)
                    
                    if not os.path.exists(pdf_file_path):
                        print(f"PDF file not found: {pdf_file_path}")
                        continue
                    
                    self._process_single_dom_file(dom_agent, dom_file_path, pdf_file_path, dom_config)
                    
                except Exception as e:
                    print(f"Error processing DOM file {dom_file}: {e}")
                    continue
            
        print("DOM image description extraction completed!")

    def _process_single_dom_file(self, dom_agent, dom_file_path, pdf_file_path, dom_config):
        """
        处理单个DOM文件
        
        Args:
            dom_agent: DomImageAgent实例
            dom_file_path: DOM文件路径
            pdf_file_path: 对应的PDF文件路径
            dom_config: DOM处理配置
        """
        print(f"Processing DOM file: {dom_file_path}")
        print(f"Corresponding PDF: {pdf_file_path}")
        
        try:
            dom_agent.model.clean_up()
            dom_data = self.load_dom_nodes(dom_file_path)
            
            # 获取文档名称和输出目录
            base_name = os.path.splitext(os.path.basename(dom_file_path))[0]
            if hasattr(self.config, 'dom_output_path'):
                base_output_dir = self.config.dom_output_path
                output_path = os.path.join(base_output_dir, f"{base_name}_with_descriptions.json")
            else:
                base_path = os.path.splitext(dom_file_path)[0]
                output_path = f"{base_path}_with_descriptions.json"
                base_output_dir = os.path.dirname(output_path)
            
            # 处理DOM元素，包括保存图像
            updated_dom_data = dom_agent.process_dom_elements(pdf_file_path, dom_data, base_name, base_output_dir)
            
            dom_agent.save_dom_with_descriptions(updated_dom_data, output_path)
            print(f"Saved updated DOM file: {output_path}")
            
            elements = updated_dom_data.get('data', {}).get('elements', [])
            figure_count = len([e for e in elements if e.get('type') == 'figure'])
            described_count = len([e for e in elements if e.get('type') == 'figure' and e.get('description')])
            
            print(f"Processed {described_count}/{figure_count} figure elements")
            
        except Exception as e:
            print(f"Error processing DOM file {dom_file_path}: {e}")
            raise

    def build_block_tree(self, block_config):
        """
        构建block tree：将DOM elements按章节层级组织成适合检索的blocks
        
        Args:
            block_config: block tree配置
        """
        # 处理所有带描述的DOM文件
        input_dir = block_config.input_path
        output_dir = block_config.output_path
        
        if not os.path.exists(input_dir):
            print(f"Input directory not found: {input_dir}")
            return
        
        # 找到所有_with_descriptions.json文件
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
                print(f"Error processing {dom_file}: {e}")
                continue
        
        print("Block tree construction completed!")

    def _process_single_dom_to_blocks(self, dom_file_path, output_path, block_config):
        """
        处理单个DOM文件，构建blocks
        
        Args:
            dom_file_path: DOM文件路径
            output_path: 输出路径
            block_config: 配置
        """
        print(f"Processing: {dom_file_path}")
        
        # 加载DOM数据
        dom_data = self.load_dom_nodes(dom_file_path)
        elements = dom_data.get('data', {}).get('elements', [])
        doc_name = os.path.splitext(os.path.basename(dom_file_path))[0].replace('_with_descriptions', '')
        
        # 分析章节层级结构
        chapter_hierarchy = self._analyze_chapter_hierarchy(elements)
        
        # 找到叶子章节
        leaf_chapters = self._find_leaf_chapters(chapter_hierarchy)
        
        # 构建blocks
        blocks = self._build_chapter_blocks(elements, leaf_chapters, chapter_hierarchy, block_config)
        
        # 处理孤立元素（parent_chapter=-1且非标题）
        orphan_blocks = self._handle_orphan_elements(elements, block_config)
        blocks.extend(orphan_blocks)
        
        # 保存结果
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
                "timestamp": self.time
            },
            "blocks": blocks
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Created {len(blocks)} blocks -> {output_path}")

    def _analyze_chapter_hierarchy(self, elements):
        """
        分析章节层级结构
        
        Args:
            elements: DOM元素列表
            
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
        """
        找到叶子章节（没有子章节的最底层章节）
        
        Args:
            chapter_hierarchy: 章节层级结构
            
        Returns:
            list: 叶子章节ID列表
        """
        leaf_chapters = []
        
        for chapter_id, info in chapter_hierarchy.items():
            if not info['children']:  # 没有子章节
                leaf_chapters.append(chapter_id)
        
        return leaf_chapters

    def _build_chapter_blocks(self, elements, leaf_chapters, chapter_hierarchy, block_config):
        """
        为所有章节构建blocks（修复版本：不再只处理叶子章节）
        
        Args:
            elements: DOM元素列表
            leaf_chapters: 叶子章节列表（保留参数兼容性，但不再使用）
            chapter_hierarchy: 章节层级结构
            block_config: 配置
            
        Returns:
            list: blocks列表
        """
        blocks = []
        
        # 修复：处理所有章节，而不是只处理叶子章节
        all_chapters = list(chapter_hierarchy.keys())
        for chapter_id in all_chapters:
            chapter_elements = self._get_chapter_elements(elements, chapter_id)
            
            if not chapter_elements:
                continue
                
            word_count = self._calculate_word_count(chapter_elements)
            
            if word_count <= block_config.max_words:
                # 创建单个block
                block = self._create_block(
                    f"chapter_{chapter_id}",
                    chapter_elements,
                    chapter_hierarchy,
                    chapter_id
                )
                blocks.append(block)
            else:
                # 分割成多个blocks
                sub_blocks = self._split_large_chapter(
                    chapter_id, 
                    chapter_elements, 
                    chapter_hierarchy,
                    block_config
                )
                blocks.extend(sub_blocks)
        
        return blocks

    def _get_chapter_elements(self, elements, chapter_id):
        """
        获取属于指定章节的直接内容元素（修复版本：避免重复收集子章节内容）
        
        Args:
            elements: DOM元素列表
            chapter_id: 章节ID
            
        Returns:
            list: 属于该章节的直接内容元素列表
        """
        # 先找到该章节的所有直接子章节ID
        child_chapter_ids = set()
        for element in elements:
            if element.get('is_chapter_title') and element.get('parent_chapter') == chapter_id:
                child_chapter_ids.add(element['index'])
        
        chapter_elements = []
        
        for element in elements:
            # 章节标题本身
            if element.get('is_chapter_title') and element['index'] == chapter_id:
                chapter_elements.append(element)
            # 属于该章节的直接内容元素（排除子章节标题，因为子章节会单独处理）
            elif (element.get('parent_chapter') == chapter_id and 
                  element['index'] not in child_chapter_ids):
                chapter_elements.append(element)
        
        # 按索引排序
        chapter_elements.sort(key=lambda x: x['index'])
        return chapter_elements

    def _calculate_word_count(self, elements):
        """
        计算元素列表的总词数（使用原始词数统计）
        
        Args:
            elements: 元素列表
            
        Returns:
            int: 总词数
        """
        total_words = 0
        
        for element in elements:
            # 计算文本词数（text + description）
            text = element.get('text', '') + ' ' + element.get('description', '')
            raw_words = len(text.split())
            total_words += raw_words
        
        return int(total_words)

    def _create_block(self, block_id, elements, chapter_hierarchy, chapter_id):
        """
        创建单个block
        
        Args:
            block_id: block ID
            elements: 元素列表
            chapter_hierarchy: 章节层级结构
            chapter_id: 章节ID
            
        Returns:
            dict: block数据
        """
        # 构建父章节路径
        parent_path = self._build_parent_path(chapter_hierarchy, chapter_id)
        
        # 统计信息
        text_count = len([e for e in elements if e.get('type') == 'paragraph'])
        figure_count = len([e for e in elements if e.get('type') == 'figure'])
        table_count = len([e for e in elements if e.get('type') == 'table'])
        
        # 页面范围
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
        """
        构建父章节路径
        
        Args:
            chapter_hierarchy: 章节层级结构
            chapter_id: 章节ID
            
        Returns:
            list: 从根到当前章节的路径
        """
        path = []
        current_id = chapter_id
        
        while current_id != -1 and current_id in chapter_hierarchy:
            path.insert(0, chapter_hierarchy[current_id]['title'])
            current_id = chapter_hierarchy[current_id]['parent']
        
        return path

    def _split_large_chapter(self, chapter_id, elements, chapter_hierarchy, block_config):
        """
        分割超长章节
        
        Args:
            chapter_id: 章节ID
            elements: 元素列表
            chapter_hierarchy: 章节层级结构
            block_config: 配置
            
        Returns:
            list: 分割后的blocks
        """
        # 获取章节标题
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
                # 创建当前block
                if len(current_elements) > 0:
                    block = self._create_block(
                        f"chapter_{chapter_id}_part_{part_num}",
                        current_elements,
                        chapter_hierarchy,
                        chapter_id
                    )
                    blocks.append(block)
                
                # 开始新block
                part_num += 1
                current_elements = [title_element] if block_config.get('include_parent_context', True) else []
                current_elements.append(element)
                current_words = self._calculate_word_count(current_elements)
        
        # 处理最后一个block
        if len(current_elements) > 0:
            block = self._create_block(
                f"chapter_{chapter_id}_part_{part_num}",
                current_elements,
                chapter_hierarchy,
                chapter_id
            )
            blocks.append(block)
        
        return blocks

    def _handle_orphan_elements(self, elements, block_config):
        """
        处理孤立元素（parent_chapter=-1且非标题）
        
        Args:
            elements: DOM元素列表
            block_config: 配置
            
        Returns:
            list: orphan blocks
        """
        orphan_elements = [
            e for e in elements 
            if e.get('parent_chapter') == -1 and not e.get('is_chapter_title')
        ]
        
        if not orphan_elements:
            return []
        
        # 按页面分组孤立元素
        page_groups = {}
        for element in orphan_elements:
            page = element.get('page', 0)
            if page not in page_groups:
                page_groups[page] = []
            page_groups[page].append(element)
        
        blocks = []
        for page, page_elements in page_groups.items():
            # 为每页的孤立元素创建block
            block = {
                "block_id": f"orphan_page_{page}",
                "block_type": "orphan_content",
                "title": f"Page {page} Content",
                "parent_path": [],
                "word_count": self._calculate_word_count(page_elements),
                "element_count": len(page_elements),
                "content_stats": {
                    "text_count": len([e for e in page_elements if e.get('type') == 'paragraph']),
                    "figure_count": len([e for e in page_elements if e.get('type') == 'figure']),
                    "table_count": len([e for e in page_elements if e.get('type') == 'table'])
                },
                "page_range": [page, page],
                "elements": page_elements
            }
            blocks.append(block)
        
        return blocks

    def load_blocks_with_mapping(self, doc_name):
        """
        加载文档的blocks并建立id映射
        
        Args:
            doc_name: 文档名称
            
        Returns:
            tuple: (blocks列表, {block_id: block_data}映射字典)
        """
        blocks_file = os.path.join(
            getattr(self.config, 'base_dir', '/tmp'),
            'tmp', 
            self.config.name, 
            'blocks', 
            f"{doc_name}_blocks.json"
        )
        
        if not os.path.exists(blocks_file):
            print(f"Blocks file not found: {blocks_file}")
            return [], {}
            
        with open(blocks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        blocks = data.get('blocks', [])
        block_mapping = {block['block_id']: block for block in blocks}
        
        return blocks, block_mapping

    def get_block_by_id(self, doc_name, block_id):
        """
        根据block_id获取单个block数据
        
        Args:
            doc_name: 文档名称
            block_id: block ID
            
        Returns:
            dict: block数据，如果未找到返回None
        """
        _, block_mapping = self.load_blocks_with_mapping(doc_name)
        return block_mapping.get(block_id)

    def get_blocks_by_ids(self, doc_name, block_ids):
        """
        批量获取多个blocks数据
        
        Args:
            doc_name: 文档名称
            block_ids: block ID列表
            
        Returns:
            list: blocks数据列表
        """
        _, block_mapping = self.load_blocks_with_mapping(doc_name)
        return [block_mapping.get(block_id) for block_id in block_ids if block_id in block_mapping]

    def dump_data(self, samples, use_retreival=True):
        if use_retreival:
            path = self.config.sample_with_retrieval_path
        else:
            path = self.config.sample_path

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(samples, f, indent = 4)
        
        return path
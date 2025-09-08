import json
import re   
from dataclasses import dataclass
import os
from tqdm import tqdm
import pymupdf
import glob
from PIL import Image
from datetime import datetime
from .block_builder import BlockBuilder


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

                        # 保持 MinerU 的原始页码索引（不做 +1）
                    
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

            dom_files = glob.glob(os.path.join(dom_dir, '**', '*content_list.json'), recursive=True)
            print(f"Found {len(dom_files)} DOM files in {dom_dir}")

            layout_files = glob.glob(os.path.join(dom_dir, '**/*layout.json'), recursive=True)

            self.enrich_content_with_layout(dom_files, layout_files)

            for dom_file in tqdm(dom_files, desc="Processing DOM files"):
                try:
                    # dom_file is already a full path; use it directly
                    dom_file_path = dom_file
                    pdf_name = os.path.basename(os.path.dirname(dom_file)) + '.pdf'
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
            # 统一 MinerU -> ChatDoc 字段格式
            dom_data = self._normalize_mineru_to_chatdoc(dom_data, dom_config)
            
            # 获取文档名称和输出目录
            base_name = os.path.splitext(os.path.basename(dom_file_path))[0]
            if base_name == 'content_list':
                base_name = os.path.basename(os.path.dirname(dom_file_path))
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
        # Delegate to dedicated builder for maintainability
        builder = BlockBuilder(self)
        builder.build_block_tree(block_config)

    

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

    def _normalize_mineru_to_chatdoc(self, dom_data, dom_config):
        """
        将 MinerU 的 content_list 结构规范化为 ChatDoc 风格：
        - 外层统一为 {"data": {"elements": [...]}}
        - 字段映射：type=image->figure, text->paragraph；page_idx/page_index/pageNo -> page(0-based)
        - 坐标：若无 outline 且存在 bbox/outline，则确保 outline 为 [x0,y0,x1,y1]
        - 填充默认：is_chapter_title=False, parent_chapter=-1, rotation=0.0, index（缺失则按序补齐）
        若 dom_method 非 mineru 或数据已是 ChatDoc 风格，直接原样返回。
        """
        try:
            if getattr(dom_config, 'dom_method', '') != 'mineru':
                return dom_data

            # 拿到元素列表
            if isinstance(dom_data, list):
                elements = dom_data
            elif isinstance(dom_data, dict):
                maybe = dom_data.get('data', {}).get('elements')
                if isinstance(maybe, list):
                    return dom_data  # 已是 ChatDoc 风格
                elements = dom_data.get('elements') if isinstance(dom_data.get('elements'), list) else []
            else:
                elements = []

            if not isinstance(elements, list) or not elements:
                return {"data": {"elements": []}}

            normalized = []
            for i, e in enumerate(elements):
                if not isinstance(e, dict):
                    continue
                ne = dict(e)

                # type 映射
                t = ne.get('type')
                if t == 'image':
                    ne['type'] = 'figure'
                elif t == 'text':
                    ne['type'] = 'paragraph'

                # 坐标标准化
                if 'outline' in ne and isinstance(ne['outline'], (list, tuple)) and len(ne['outline']) == 4:
                    ne['outline'] = list(ne['outline'])
                elif 'bbox' in ne and isinstance(ne['bbox'], (list, tuple)) and len(ne['bbox']) == 4:
                    ne['outline'] = list(ne['bbox'])

                # 页码统一到 0-based 的 page
                if 'page' not in ne:
                    if 'page_idx' in ne:
                        try:
                            ne['page'] = max(int(ne['page_idx']), 0)
                        except Exception:
                            ne['page'] = 0
                    elif 'page_index' in ne:
                        try:
                            ne['page'] = int(ne['page_index'])
                        except Exception:
                            ne['page'] = 0
                    elif 'pageNo' in ne:
                        try:
                            ne['page'] = int(ne['pageNo'])
                        except Exception:
                            ne['page'] = 0

                # 默认值
                ne.setdefault('is_chapter_title', False)
                ne.setdefault('parent_chapter', -1)
                ne.setdefault('rotation', 0.0)
                ne.setdefault('index', i)

                # figure 的 text 补充 image_caption
                if ne.get('type') == 'figure' and 'text' not in ne:
                    cap = ne.get('image_caption')
                    if isinstance(cap, list):
                        ne['text'] = '\n'.join([str(c) for c in cap if c is not None])
                    elif isinstance(cap, str):
                        ne['text'] = cap
                    else:
                        ne['text'] = ''

                normalized.append(ne)

            return {"data": {"elements": normalized}}
        except Exception:
            return dom_data

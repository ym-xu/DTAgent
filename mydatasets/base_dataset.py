import json
import re   
from dataclasses import dataclass
import os
from tqdm import tqdm
import pymupdf
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
        if hasattr(dom_config, 'dom_files'):
            dom_files = dom_config.dom_files
            pdf_files = dom_config.pdf_files if hasattr(dom_config, 'pdf_files') else []
            
            if len(dom_files) != len(pdf_files):
                print("Error: Number of DOM files and PDF files must match")
                return
            
            for dom_file, pdf_file in zip(dom_files, pdf_files):
                self._process_single_dom_file(dom_agent, dom_file, pdf_file, dom_config)
        else:
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

    def dump_data(self, samples, use_retreival=True):
        if use_retreival:
            path = self.config.sample_with_retrieval_path
        else:
            path = self.config.sample_path

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(samples, f, indent = 4)
        
        return path
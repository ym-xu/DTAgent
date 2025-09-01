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



    def dump_data(self, samples, use_retreival=True):
        if use_retreival:
            path = self.config.sample_with_retrieval_path
        else:
            path = self.config.sample_path

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(samples, f, indent = 4)
        
        return path
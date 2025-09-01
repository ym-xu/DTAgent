from agents.base_agent import Agent
import pymupdf
from PIL import Image
import tempfile
import os
import json
from typing import List, Dict, Any

class DomImageAgent(Agent):
    def __init__(self, config, model=None):
        super().__init__(config, model)
        self.system_prompt = config.agent.system_prompt if hasattr(config.agent, 'system_prompt') else ""

    def crop_image_from_pdf(self, pdf_path: str, page_num: int, outline: List[float]) -> str:
        """
        从PDF中裁剪指定区域的图像
        
        Args:
            pdf_path: PDF文件路径
            page_num: 页码（从0开始）
            outline: 坐标 [x1, y1, x2, y2]
            
        Returns:
            裁剪后图像的临时文件路径
        """
        with pymupdf.open(pdf_path) as pdf:
            if page_num >= len(pdf):
                raise ValueError(f"Page {page_num} does not exist in PDF")
            
            page = pdf[page_num]
            # 创建裁剪矩形
            rect = pymupdf.Rect(outline[0], outline[1], outline[2], outline[3])
            
            # 获取页面像素图
            pix = page.get_pixmap(clip=rect, dpi=150)
            
            # 创建临时文件
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            pix.save(temp_file.name)
            temp_file.close()
            
            return temp_file.name

    def describe_dom_image(self, pdf_path: str, element: Dict[str, Any]) -> str:
        """
        为DOM元素中的图像生成描述
        
        Args:
            pdf_path: PDF文件路径
            element: DOM元素字典，包含type、page、outline等信息
            
        Returns:
            图像描述文本
        """
        if element.get('type') != 'figure':
            return ""
        
        try:
            # 从PDF中裁剪图像
            page_num = element.get('page', 0)
            outline = element.get('outline', [])
            
            if len(outline) != 4:
                return "Invalid outline coordinates"
            
            # 裁剪图像
            temp_image_path = self.crop_image_from_pdf(pdf_path, page_num, outline)
            
            # 构建问题
            question = self.system_prompt + "\n\nPlease describe this image:"
            
            # 使用模型生成描述
            try:
                description, _ = self.model.predict(
                    question=question,
                    images=[temp_image_path]
                )
                
                # 清理临时文件
                os.unlink(temp_image_path)
                
                return description.strip()
                
            except Exception as e:
                # 清理临时文件
                if os.path.exists(temp_image_path):
                    os.unlink(temp_image_path)
                return f"Error generating description: {str(e)}"
                
        except Exception as e:
            return f"Error processing image: {str(e)}"

    def process_dom_elements(self, pdf_path: str, dom_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理DOM数据中的所有图像元素，为其添加描述
        
        Args:
            pdf_path: PDF文件路径
            dom_data: DOM数据字典
            
        Returns:
            更新后的DOM数据字典
        """
        elements = dom_data.get('elements', [])
        
        for element in elements:
            if element.get('type') == 'figure' and not element.get('description'):
                print(f"Processing image element {element.get('index', 'unknown')} on page {element.get('page', 'unknown')}")
                
                description = self.describe_dom_image(pdf_path, element)
                element['description'] = description
                
                # 清理GPU内存
                self.model.clean_up()
        
        return dom_data

    def save_dom_with_descriptions(self, dom_data: Dict[str, Any], output_path: str):
        """
        保存添加了描述的DOM数据
        
        Args:
            dom_data: DOM数据字典
            output_path: 输出文件路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dom_data, f, indent=2, ensure_ascii=False)
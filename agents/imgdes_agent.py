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

    def get_full_page_image_from_pdf(self, pdf_path: str, page_num: int) -> str:
        """
        从PDF中获取完整页面图像
        
        Args:
            pdf_path: PDF文件路径
            page_num: 页码（从0开始）
            
        Returns:
            完整页面图像的临时文件路径
        """
        with pymupdf.open(pdf_path) as pdf:
            if page_num >= len(pdf):
                raise ValueError(f"Page {page_num} does not exist in PDF")
            
            page = pdf[page_num]
            
            # 获取完整页面像素图
            pix = page.get_pixmap(dpi=150)
            
            # 创建临时文件
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            pix.save(temp_file.name)
            temp_file.close()
            
            return temp_file.name

    def save_cropped_image(self, pdf_path: str, page_num: int, outline: List[float], doc_name: str, element_index: int, base_output_dir: str) -> str:
        """
        保存裁剪后的图像到指定目录
        
        Args:
            pdf_path: PDF文件路径
            page_num: 页码（从0开始）
            outline: 坐标 [x1, y1, x2, y2]
            doc_name: 文档名称（用于创建子目录）
            element_index: 元素索引，用于文件命名
            base_output_dir: 基础输出目录（dom_output_path）
            
        Returns:
            保存的图像文件相对路径
        """
        with pymupdf.open(pdf_path) as pdf:
            if page_num >= len(pdf):
                raise ValueError(f"Page {page_num} does not exist in PDF")
            
            page = pdf[page_num]
            # 创建裁剪矩形
            rect = pymupdf.Rect(outline[0], outline[1], outline[2], outline[3])
            
            # 获取裁剪区域像素图
            pix = page.get_pixmap(clip=rect, dpi=150)
            
            # 创建文档专用的图片目录
            image_dir = os.path.join(base_output_dir, os.path.basename(doc_name))
            os.makedirs(image_dir, exist_ok=True)
            
            # 生成文件名：page_{page_num}_element_{element_index}.png
            filename = f"page_{page_num}_element_{element_index}.png"
            output_path = os.path.join(image_dir, filename)
            
            # 保存图像
            pix.save(output_path)
            
            # 返回相对路径（相对于base_output_dir）
            relative_path = os.path.join(os.path.basename(doc_name), filename)
            
            return relative_path

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
            # 获取页面和坐标信息
            page_num = element.get('page', 0)
            outline = element.get('outline', [])
            
            if len(outline) != 4:
                return "Invalid outline coordinates"
            
            # 获取完整页面图像
            temp_image_path = self.get_full_page_image_from_pdf(pdf_path, page_num)
            
            # 构建问题，填入坐标信息
            x1, y1, x2, y2 = outline
            question = self.system_prompt.format(x1=x1, y1=y1, x2=x2, y2=y2)
            
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

    def process_dom_elements(self, pdf_path: str, dom_data: Dict[str, Any], doc_name: str = None, base_output_dir: str = None) -> Dict[str, Any]:
        """
        处理DOM数据中的所有图像元素，为其添加描述和保存图像
        
        Args:
            pdf_path: PDF文件路径
            dom_data: DOM数据字典
            doc_name: 文档名称（用于图像保存）
            base_output_dir: 基础输出目录（用于图像保存）
            
        Returns:
            更新后的DOM数据字典
        """
        elements = dom_data.get('data', {}).get('elements', [])
        for element in elements:
            if element.get('type') == 'figure' and not element.get('description'):
                # 检查图像大小，跳过太小的图像
                outline = element.get('outline', [])
                if len(outline) >= 4:
                    width = outline[2] - outline[0]
                    height = outline[3] - outline[1]
                    min_size = getattr(self.config.agent, 'min_image_size', 50)  # 默认最小尺寸50像素
                    
                    if width < min_size or height < min_size:
                        print(f"Skipping small image element {element.get('index', 'unknown')} (size: {width}x{height})")
                        continue
                
                # print(f"Processing image element {element.get('index', 'unknown')} on page {element.get('page', 'unknown')}")
                
                # 生成描述
                description = self.describe_dom_image(pdf_path, element)
                element['description'] = description
                
                # 保存图像并添加链接
                if doc_name and base_output_dir:
                    try:
                        page_num = element.get('page', 0)
                        element_index = element.get('index', 0)
                        
                        image_path = self.save_cropped_image(
                            pdf_path, page_num, outline, doc_name, element_index, base_output_dir
                        )
                        element['image_link'] = image_path
                        
                    except Exception as e:
                        print(f"Error saving image for element {element.get('index', 'unknown')}: {e}")
                        element['image_link'] = None
                
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
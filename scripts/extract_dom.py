import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mydatasets.base_dataset import BaseDataset
from agents.dom_image_agent import DomImageAgent
import hydra
import importlib
from tqdm import tqdm

@hydra.main(config_path="../config", config_name="mmlb", version_base="1.2")
def main(cfg):
    """
    DOM图像描述提取脚本
    从DOM JSON文件中的图像元素生成描述
    """
    print("Starting DOM image description extraction...")
    print(f"Configuration: {cfg}")
    
    # 检查是否启用DOM图像描述功能
    if not cfg.get('enable_dom_image_description', False):
        print("DOM image description is disabled. Set 'enable_dom_image_description: true' to enable.")
        return
    
    # 设置CUDA设备
    if hasattr(cfg, 'dom_image_description') and hasattr(cfg.dom_image_description, 'cuda_visible_devices'):
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.dom_image_description.cuda_visible_devices
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    
    # 创建数据集实例
    dataset = BaseDataset(cfg.dataset)
    
    # 创建DOM图像描述智能体
    if not hasattr(cfg, 'dom_image_description'):
        print("Error: dom_image_description configuration not found")
        return
    
    # 动态加载智能体和模型配置
    agent_name = cfg.dom_image_description.agent
    model_name = cfg.dom_image_description.model
    
    agent_cfg = hydra.compose(config_name=f"agent/{agent_name}", overrides=[]).agent
    model_cfg = hydra.compose(config_name=f"model/{model_name}", overrides=[]).model
    
    # 创建模型实例
    module = importlib.import_module(model_cfg.module_name)
    model_class = getattr(module, model_cfg.class_name)
    model = model_class(model_cfg)
    print(f"Created model: {model_cfg.class_name}")
    
    # 创建智能体实例
    from omegaconf import OmegaConf
    agent_config = OmegaConf.create({
        'agent': agent_cfg,
        'model': model_cfg
    })
    
    dom_agent = DomImageAgent(agent_config, model)
    print(f"Created agent: {agent_name}")
    
    # 获取需要处理的文件列表
    if hasattr(cfg.dom_image_description, 'dom_files'):
        # 处理指定的DOM文件列表
        dom_files = cfg.dom_image_description.dom_files
        pdf_files = cfg.dom_image_description.pdf_files if hasattr(cfg.dom_image_description, 'pdf_files') else []
        
        if len(dom_files) != len(pdf_files):
            print("Error: Number of DOM files and PDF files must match")
            return
        
        for dom_file, pdf_file in zip(dom_files, pdf_files):
            process_single_dom_file(dom_agent, dom_file, pdf_file, cfg)
    
    else:
        # 处理数据集中的所有DOM文件
        dom_dir = cfg.dataset.dom_path
        if not os.path.exists(dom_dir):
            print(f"DOM directory not found: {dom_dir}")
            return
        
        # 查找所有JSON文件
        dom_files = [f for f in os.listdir(dom_dir) if f.endswith('.json')]
        print(f"Found {len(dom_files)} DOM files in {dom_dir}")
        
        for dom_file in tqdm(dom_files, desc="Processing DOM files"):
            try:
                dom_file_path = os.path.join(dom_dir, dom_file)
                
                # 推断对应的PDF文件名
                pdf_name = dom_file.replace('.json', '.pdf')
                pdf_file_path = os.path.join(cfg.dataset.document_path, pdf_name)
                
                if not os.path.exists(pdf_file_path):
                    print(f"PDF file not found: {pdf_file_path}")
                    continue
                
                # 处理DOM文件
                process_single_dom_file(dom_agent, dom_file_path, pdf_file_path, cfg)
                
            except Exception as e:
                print(f"Error processing DOM file {dom_file}: {e}")
                continue
    
    print("DOM image description extraction completed!")

def process_single_dom_file(dom_agent, dom_file_path, pdf_file_path, cfg):
    """
    处理单个DOM文件
    
    Args:
        dom_agent: DOM图像描述智能体实例
        dom_file_path: DOM JSON文件路径
        pdf_file_path: 对应的PDF文件路径
        cfg: 配置对象
    """
    print(f"Processing DOM file: {dom_file_path}")
    print(f"Corresponding PDF: {pdf_file_path}")
    
    try:
        # 清理内存
        dom_agent.model.clean_up()
        
        # 使用数据集类的方法加载DOM数据
        dataset = BaseDataset(cfg.dataset)
        dom_data = dataset.load_dom_nodes(dom_file_path)
        
        # 处理DOM中的图像元素
        updated_dom_data = dom_agent.process_dom_elements(pdf_file_path, dom_data)
        
        # 确定输出路径
        if hasattr(cfg.dataset, 'dom_output_path'):
            base_name = os.path.splitext(os.path.basename(dom_file_path))[0]
            output_path = os.path.join(cfg.dataset.dom_output_path, f"{base_name}_with_descriptions.json")
        else:
            # 默认在原文件目录下创建新文件
            base_path = os.path.splitext(dom_file_path)[0]
            output_path = f"{base_path}_with_descriptions.json"
        
        # 保存更新后的DOM数据
        dom_agent.save_dom_with_descriptions(updated_dom_data, output_path)
        print(f"Saved updated DOM file: {output_path}")
        
        # 统计处理结果
        elements = updated_dom_data.get('elements', [])
        figure_count = len([e for e in elements if e.get('type') == 'figure'])
        described_count = len([e for e in elements if e.get('type') == 'figure' and e.get('description')])
        
        print(f"Processed {described_count}/{figure_count} figure elements")
        
    except Exception as e:
        print(f"Error processing DOM file {dom_file_path}: {e}")
        raise

if __name__ == "__main__":
    main()
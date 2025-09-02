import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mydatasets.base_dataset import BaseDataset
from agents.multi_agent_system import MultiAgentSystem
import hydra

@hydra.main(config_path="../config", config_name="mmlb", version_base="1.2")
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.dom_image_description.cuda_visible_devices
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    
    # 动态加载agent和model配置（参考predict.py的做法）
    for agent_config in cfg.dom_image_description.agents:
        agent_name = agent_config.agent
        model_name = agent_config.model
        agent_cfg = hydra.compose(config_name="agent/"+agent_name, overrides=[]).agent
        model_cfg = hydra.compose(config_name="model/"+model_name, overrides=[]).model
        agent_config.agent = agent_cfg
        agent_config.model = model_cfg
    
    multi_system = MultiAgentSystem(cfg.dom_image_description)
    dom_agent = multi_system.agents[0]
    
    dataset = BaseDataset(cfg.dataset)
    dataset.process_dom_images(dom_agent, cfg.dom_image_description)

if __name__ == "__main__":
    main()
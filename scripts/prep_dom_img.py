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
    
    multi_system = MultiAgentSystem(cfg.dom_image_description)
    dom_agent = multi_system.agents[0]
    
    dataset = BaseDataset(cfg.dataset)
    dataset.process_dom_images(dom_agent, cfg.dom_image_description)

if __name__ == "__main__":
    main()
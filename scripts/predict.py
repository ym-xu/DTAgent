import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mydatasets.base_dataset import BaseDataset
from agents.mdoc_agent import MDocAgent
import hydra

@hydra.main(config_path="../config", config_name="base", version_base="1.2")
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.mdoc_agent.cuda_visible_devices
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    for agent_config in cfg.mdoc_agent.agents:
        agent_name = agent_config.agent
        model_name = agent_config.model
        agent_cfg = hydra.compose(config_name="agent/"+agent_name, overrides=[]).agent
        model_cfg = hydra.compose(config_name="model/"+model_name, overrides=[]).model
        agent_config.agent = agent_cfg
        agent_config.model = model_cfg

    dataset = BaseDataset(cfg.dataset)
    # print(cfg.mdoc_agent.agents)
    mdoc_agent = MDocAgent(cfg.mdoc_agent)

if __name__ == "__main__":
    main()
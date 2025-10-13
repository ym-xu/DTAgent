from tqdm import tqdm
import importlib
import json
import torch
import os
from agents.multi_agent_system import MultiAgentSystem
from agents.base_agent import Agent
from mydatasets.base_dataset import BaseDataset

class MDocAgent(MultiAgentSystem):
    def __init__(self, config):
        super().__init__(config)
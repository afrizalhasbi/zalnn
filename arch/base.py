import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
import json
import os

class BaseArch(nn.Module):
    def __init__(self, config=None):
        super(BaseArch, self).__init__()
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device=self.device, dtype=torch.bfloat16)
        
    def num_params(self):
        num_params = sum(p.numel() for p in self.parameters())
        return num_params

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        state_dict = self.state_dict()
        save_file(state_dict, f"{path}/model.safetensors")
        with open (f"{path}/config.json", "w") as file:
            json.dump(self.config, file, indent=2)

    # def save(self, path):
        # os.makedirs(path, exist_ok=True)
        # state_dict = self.state_dict()
        # # Transpose weights for linear and conv layers
        # transposed_dict = {
            # k: v.T if (k.endswith('.weight') and len(v.shape) == 2) else v
            # for k, v in state_dict.items()
        # }
        # save_file(transposed_dict, f"{path}/model.safetensors")
        # with open(f"{path}/config.json", "w") as file:
            # json.dump(self.config, file, indent=2)

    @classmethod
    def from_pretrained(cls, path):
        assert os.path.isdir(path), "argument `path` must be a directory" 
        print(f"Loading pretrained model from {path}")
        with open (f"{path}/config.json", "r") as file:
           config = json.load(file)
        model = cls(config).load_safetensors(path)
        return model

    def load_safetensors(self, path):
        state_dict = load_file(f"{path}/model.safetensors")
        state_dict = {k: v.to(device=self.device, dtype=torch.bfloat16) for k, v in state_dict.items()}
        self.load_state_dict(state_dict)
        return self

    @staticmethod
    def get_activation(activation):
        return {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leakyrelu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
        }.get(activation_str.lower(), nn.ReLU())

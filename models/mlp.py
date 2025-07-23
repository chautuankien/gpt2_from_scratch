import torch.nn as nn
from config.config import GPTConfig

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):  
        super().__init__()
        self.c_fc = nn.Linear(config.N_EMBED, 4 * config.N_EMBED)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.N_EMBED, config.N_EMBED)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x
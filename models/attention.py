import torch.nn as nn
from config.config import GPTConfig

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.n_head = config.n_head  # -> n_head attention
        self.n_embd = config.n_embd
         
    def forward(self, x):
        B, T, C = x.size() # B: batch size, T: sequence length, C: embedding dimension
        # calculate query, key, value matrices
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, C // n_head)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, C // n_head)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, C // n_head)

        # attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, n_head, T, T)
        # MASKING
        att = att.masked_fill(torch.tril(torch.ones(T, T, device=att.device)) == 0, float('-inf'))  # causal mask
        # END MASKING
        att = F.softmax(att, dim=-1)  # (B, n_head, T, T)
        y = att @ v # (B, n_head, T, C // n_head)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.c_proj(y)
        return y
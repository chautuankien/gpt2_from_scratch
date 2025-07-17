import torch.nn as nn

from config.config import GPTConfig
from models.attention import CausalSelfAttention
from models.mlp import MLP

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super(Block, self).__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super(GPT, self).__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # positional embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # transformer blocks
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # final layer norm
		    
		    # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
		    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # (T,)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)
        
        # forward the block of the transformer
        for block in self.transformer.h:
            x = block(x)
        
        # apply final layer norm
        x = self.transformer.ln_f(x) # (B, T, n_embd)
        
        # project to vocab size
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        # compute loss if training targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
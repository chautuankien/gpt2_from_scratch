import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect

from config.config import GPTConfig
from models.attention import CausalSelfAttention
from models.mlp import MLP

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super(Block, self).__init__()

        self.ln_1 = nn.LayerNorm(config.N_EMBED)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.N_EMBED)
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
            wte = nn.Embedding(config.VOCAB_SIZE, config.N_EMBED), # token embeddings
            wpe = nn.Embedding(config.BLOCK_SIZE, config.N_EMBED), # positional embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.N_LAYERS)]), # transformer blocks
            ln_f = nn.LayerNorm(config.N_EMBED)
        ))
        self.lm_head = nn.Linear(config.N_EMBED, config.VOCAB_SIZE, bias=False) # final layer norm
		    
	    # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.N_LAYERS) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
		    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.BLOCK_SIZE, f"Cannot forward sequence of length {T}, block size is only {self.config.BLOCK_SIZE}"
        
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
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
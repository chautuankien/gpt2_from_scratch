from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 512   # max sequence length
    vocab_size: int = 50257 # number of tokens in the vocabulary, for gpt2 tokenizer it is 50257
    n_layer: int = 12   # number of transformer blocks
    n_head: int = 12    # number of attention heads for each transformer block
    n_embd: int = 768   # embedding dimension for each tokens
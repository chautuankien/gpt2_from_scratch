from dataclasses import dataclass


@dataclass
class GPTConfig:
    BLOCK_SIZE: int = 512       # max sequence length
    VOCAB_SIZE: int = 50304     # number of tokens in the vocabulary, for gpt2 tokenizer it is 50257
    N_LAYERS: int = 12         # number of transformer blocks
    N_HEAD: int = 12           # number of attention heads for each transformer block
    N_EMBED: int = 768          # embedding dimension for each tokens

    # Paths to training and development datasets
    TRAIN_PATH: str = "data/train/wikitext-2-raw-v1.h5"  # file path for the training dataset
    VAL_PATH: str = ""   # file path for the validation dataset

    # Training parameters
    BATCH_SIZE: int = 8                 # number of sequences per training batch
    TOTAL_BATCH_SIZE: int = 32768       # effective batch size in tokens
    GRADIENT_ACCUMULATION_STEPS: int = TOTAL_BATCH_SIZE // (TOTAL_BATCH_SIZE * BLOCK_SIZE)    # accumulate gradients over this many steps
    TRAINING_STEPS: int = 50           # total number of training step
    EVAL_STEPS: int = 10                # total number of eval step
    EVAL_INTERVAL: int = 5              # total number of iterations to evaluate the model
    LOG_INTERVAL: int = 1              # frequency to perform evaluation
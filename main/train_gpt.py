import torch
import time
import math
from tqdm import tqdm

from models.transformer import GPT
from config.config import GPTConfig
from main.data_loader import get_batch_iterator, BatchIterator

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f"Using device: {device}")

torch.set_float32_matmul_precision('high')

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

model = GPT(GPTConfig())
model.to(device)

# Print the total number of params
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in the model: {total_params}")

# SETUP LR DECAY SCHEDULER
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# OPTIMIZATION
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

# TRAINING LOOP
batch_iterator = BatchIterator(config=GPTConfig(), device=device)

# List to track loss values during training.
losses = []

# Helper function to estimate average loss for training and val data
@torch.no_grad()
def estimate_loss(steps: int):
    out = {}
    model.eval()    # set model to evaluation mode

    for split in ['train', 'val']:
        # Select the appropriate data path for the current split.
        data_path = GPTConfig.TRAIN_PATH if split == 'train' else GPTConfig.VAL_PATH

        # Create batch iterator for evaluation
        batch_iterator_eval = BatchIterator(config=GPTConfig(), device=device)

        # Initialize a tensor to track loss values for each evaluation step.
        losses_eval = torch.zeros(steps)
        try:
            for k in range(steps):
                # Fetch a batch and calculate the loss
                xb, yb = next(batch_iterator_eval)
                _, loss = model(xb, yb)
                losses_eval[k] = loss.item()
        finally:
            batch_iterator.close()
        # Compute the mean loss for the current split.
        out[split] = losses_eval[:k + 1].mean()

    model.train()  # Restore the model to training mode.
    return out
"""
# pbar = tqdm(range(GPTConfig.TRAINING_STEPS))
try:
    for step, (xb, yb) in enumerate(batch_iterator):
    # for step in range(GPTConfig.TRAINING_STEPS):
        # Training logic here
        if step > GPTConfig.TRAINING_STEPS:
            break
        t0 = time.time()
        optimizer.zero_grad()   

        for micro_step in range(GPTConfig.GRADIENT_ACCUMULATION_STEPS):
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(xb, yb)  # forward pass

            # Record the loss for tracking.
            losses.append(loss.item())

            # Backpropagate the loss
            loss.backward()

        # Gradient clipping
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  

        # Determine and set learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Update model parameters
        optimizer.step()

        # Timing and Logging
        t1 = time.time()
        dt = t1 - t0
        if step % GPTConfig.EVAL_STEPS == 0:
            print(f"Step: {step}, loss: {loss.item()}")
            # losses = estimate_loss(GPTConfig.EVAL_INTERVAL)
            # print(f"Step: {step}, Train loss: {losses['train']:.4f}, Dev loss: {dlosses['val']:.4f}")
            # print(f"Step: {step}, Train loss: {loss:.6f}, Lr: {lr:.4e} Norm: {norm:.4f}, Time: {dt:.2} s")
        
finally:
    batch_iterator.close()
"""
try:
    for step in range(GPTConfig.TRAINING_STEPS):
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0.0  # reset loss accumulation   

        for micro_step in range(GPTConfig.GRADIENT_ACCUMULATION_STEPS):
            xb, yb = next(batch_iterator)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(xb, yb)  # forward pass

            loss = loss / GPTConfig.GRADIENT_ACCUMULATION_STEPS
            loss_accum += loss.detach()  # accumulate loss

            # Backpropagate the loss
            loss.backward()
        
        # Record the loss for tracking.
        losses.append(loss_accum)

        # Gradient clipping
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  

        # Determine and set learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Update model parameters
        optimizer.step()

        # Timing and Logging
        t1 = time.time()
        dt = (t1 - t0) * 1000
        if step % GPTConfig.EVAL_STEPS == 0:
            # print(f"Step: {step}, loss: {loss.item()}")
            print(f"Step: {step}, Loss: {loss_accum:.6f}, Lr: {lr:.4e} Norm: {norm:.4f}, Time: {dt:.2} ms")
            est_loss = estimate_loss(GPTConfig.EVAL_INTERVAL)
            print(f"Step: {step}, Train loss: {est_loss['train']:.4f}, Dev loss: {est_loss['val']:.4f}")

        
finally:
    batch_iterator.close()
import random
import torch, time

from dataclasses import dataclass
# from gpt_model import GPT
from recursive_model import GPT
from create_dataset import PAD_TOKEN, vocab_size, decode, encode, itos, BLOCK_SIZE


def get_ground_truth_string(y_row, itos):
    valid_indices = y_row[y_row != -100].tolist()
    return ''.join([itos[i] for i in valid_indices])


def evaluate(model, val_data, max_samples=50):
    model.eval()
    X_val, Y_val = val_data

    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        # x, y = train_loader.next_batch()
        x, y = dataset['val']
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)

        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
    
    indices = list(range(len(X_val)))
        
    num_correct = 0
    total = 0
    
    EQ_TOKEN_ID = encode('=')[0]
    
    with torch.no_grad():
        for i in indices:
            x_row = X_val[i].to(device)
            eq_idxs = (x_row == EQ_TOKEN_ID).nonzero(as_tuple=True)[0]
            if len(eq_idxs) == 0: continue
            cut_idx = eq_idxs[0].item()
            
            prompt = x_row[:cut_idx+1].unsqueeze(0) # Shape (1, T)
            
            y_row = Y_val[i]
            valid_indices = y_row[y_row != -100].tolist()
            truth_str = decode(valid_indices) # e.g. "10~"
            truth_str = truth_str.replace(PAD_TOKEN, '')
        
            completion = model.generate(prompt, max_new_tokens=10)[0]
            completion_str = decode(completion.tolist())
            
            prompt_str = decode(prompt[0].tolist())
            if prompt_str in completion_str:
                generated_answer = completion_str.split(prompt_str, 1)[1]
                generated_answer = generated_answer.split('~')[0] + '~' 
            else:
                generated_answer = ""
            if generated_answer == truth_str:
                num_correct += 1
                
            total += 1
            
            if total <= 3:
                print(f"  Prop: {prompt_str} | Truth: {truth_str} | Gen: {generated_answer}")

    acc = num_correct / total
    # print(f"Validation Accuracy: {acc*100:.2f}% ({num_correct}/{total})")
    model.train()
    return loss_accum.item(), acc


@dataclass
class GPTConfig:
    block_size: int = BLOCK_SIZE
    vocab_size: int = vocab_size
    n_embd: int = 32
    n_head: int = 4
    n_layer: int = 1
    n_recursion: int = 1
    steps: int = 1000
    effective_depth: int = n_layer * n_recursion
    bias: bool = False
    weight_decay: float = 0.2
    lr: float = 3e-4
    dataset: str = 'addition_dataset_10k_2dig.pt'


micro_batch_size = 16 # micro batch size
grad_accum_steps = 4
total_batch_size = micro_batch_size * grad_accum_steps
batch_size_tokens = total_batch_size * GPTConfig.block_size  # 524288 - number of tokens 2**19

# assert total_batch_size % (micro_batch_size * GPTConfig.block_size) == 0, "make sure total_batch_size is divisible by micro_batch_size * block_size"
# grad_accum_steps = total_batch_size // (micro_batch_size * GPTConfig.block_size)
print(f"total batch size: {total_batch_size:,} batch")
print(f"gradient accumulation steps: {grad_accum_steps}")
print(f'{batch_size_tokens:,} tokens per batch')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on {device}...')

torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig())
model.to(device)

params = sum(p.numel() for p in model.parameters())
print(f'Model has {params:,} parameters.')

compiled = False
if device == 'cuda' and False:
    print(f'Compiling model...')
    model = torch.compile(model)
    compiled = True
else:
    print(f'Device not cuda, not compiling.')

print(f'Loading dataset...')
dataset = torch.load(GPTConfig.dataset)

def get_lr(step):
    return GPTConfig.lr


optimizer = torch.optim.AdamW(model.parameters(), lr=GPTConfig.lr, weight_decay=GPTConfig.weight_decay)

print(f'Beginning training...')
for step in range(GPTConfig.steps):
    t0 = time.time()
    optimizer.zero_grad()

    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        # x, y = train_loader.next_batch()
        x, y = dataset['train']
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)

        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # lr scheduling here
    lr = get_lr(step)

    optimizer.step()
    if device == 'cuda':  # and compiled ?
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)

    tokens_processed = total_batch_size * GPTConfig.block_size
    tokens_per_sec = tokens_processed / dt

    if ((step+1) % 100 == 0) or (step+1) == GPTConfig.steps:
        val_loss, val_acc = evaluate(model, dataset['val'])
        print(f"step {step+1:4d} | t/loss: {loss_accum.item():.4f} | val/loss: {val_loss:.4f} | val/acc: {val_acc:.2f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")


# code inference, eval, grad_accum
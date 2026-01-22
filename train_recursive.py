import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from torch.optim.lr_scheduler import LambdaLR

from recursive_model import GPT, GPTConfig 

from create_dataset import vocab_size, BLOCK_SIZE, PAD_TOKEN_ID, decode

@dataclass
class TrainConfig:
    block_size: int = BLOCK_SIZE
    vocab_size: int = vocab_size
    n_embd: int = 32
    n_head: int = 4
    n_layer: int = 1
    
    # recursive params
    num_register_tokens: int = 0
    num_latent_refinements: int = 2
    num_output_refinements: int = 1
    max_recurrent_steps: int = 4
    
    # training params
    max_steps: int = 20000
    batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.1
    halt_loss_weight: float = 1.0
    halt_prob_threshold: float = 0.5
    
    # system
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_path: str = 'addition_dataset_10k_2dig.pt'

config = TrainConfig()


@torch.no_grad()
def evaluate_accuracy(model, dataset, config, decode, max_batches=50, print_examples=True):
    model.eval()
    
    total_correct = 0
    total_samples = 0
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    
    # --- PART 1: Calculate Score (Fast) ---
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
            
        x, y = x.to(config.device), y.to(config.device)
        
        # Sanitize input
        x_clean = x.clone()
        x_clean[x_clean == -100] = 0
        
        # Fast "Parallel" Prediction (Teacher Forcing)
        # Checks: "If I gave you the context, would you have guessed right?"
        preds, _ = model.predict(
            x_clean, 
            halt_threshold=config.halt_prob_threshold, 
            max_deep_refinement_steps=config.max_recurrent_steps
        )
        
        mask = (y != -100)
        correct_tokens = (preds == y) | (~mask)
        is_sequence_correct = correct_tokens.all(dim=1)
        
        total_correct += is_sequence_correct.sum().item()
        total_samples += x.shape[0]

    # --- PART 2: Visual Examples (Real Generation) ---
    if print_examples:
        print("\n" + "="*50)
        print("Real Generation Examples (No Teacher Forcing)")
        print("="*50)
        
        # Pick the first 3 examples from the last batch processed
        # We need to find where the prompt ends (the first -100 in target y)
        
        for j in range(min(3, x.shape[0])):
            curr_x = x[j]
            curr_y = y[j]
            
            # Find the split point: where does the target answer start?
            # In your dataset, x is full seq, y is target. 
            # Prompt is where y is -100.
            prompt_mask = (curr_y == -100)
            
            # Extract just the prompt tokens (e.g. "12 + 15 =")
            # We assume padding (0) might be at the start, we want to keep it or handle it
            # Usually we just take the valid tokens that are NOT part of the answer
            prompt_tokens = curr_x[prompt_mask]
            
            # Remove padding 0s if they exist and are just placeholders
            prompt_tokens = prompt_tokens[prompt_tokens != 0]
            
            if len(prompt_tokens) == 0: continue

            # Reshape for generate: (1, SeqLen)
            prompt_tensor = prompt_tokens.unsqueeze(0).to(config.device)
            
            # Generate! 
            # (Generates 5 new tokens - adjust based on your dataset answer length)
            generated_full = model.generate(
                prompt_tensor, 
                max_new_tokens=4, 
                halt_threshold=config.halt_prob_threshold,
                max_outer_steps=config.max_recurrent_steps
            )
            
            # Decode
            prompt_str = decode(prompt_tokens.tolist())
            gen_only = generated_full[0, len(prompt_tokens):] # Only the new stuff
            gen_str = decode(gen_only.tolist())
            
            # Get Ground Truth for comparison
            truth_tokens = curr_y[~prompt_mask]
            truth_str = decode(truth_tokens.tolist())
            
            # Visual Marker
            is_correct = (gen_str.strip() == truth_str.strip())
            marker = "✅" if is_correct else "❌"
            
            print(f"Prompt: {prompt_str}")
            print(f"Truth:  {truth_str}")
            print(f"Gen:    {gen_str} {marker}")
            print("-" * 30)
        print("\n")

    model.train()
    return total_correct / total_samples


def get_batch(dataset, batch_size, device):
    ix = torch.randint(len(dataset), (batch_size,))
    x = torch.stack([dataset[i][0] for i in ix])
    y = torch.stack([dataset[i][1] for i in ix])
    return x.to(device), y.to(device)


def get_lr_schedule(step, warmup_steps=1000, max_steps=5000):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    return 1.0


print(f"Running on {config.device}...")

gpt_config = GPTConfig(
    block_size=config.block_size,
    vocab_size=config.vocab_size,
    n_embd=config.n_embd,
    n_head=config.n_head,
    n_layer=config.n_layer,

    # recursive configs
    num_register_tokens=config.num_register_tokens,
    num_latent_refinements=config.num_latent_refinements,
    num_output_refinements=config.num_output_refinements,
    halt_loss_weight=config.halt_loss_weight
)

model = GPT(gpt_config)
model.to(config.device)

# TODO: change to MuonAdamAtan2 ?
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate / (config.batch_size * config.max_recurrent_steps),
    weight_decay=config.weight_decay
)

print(f"Loading dataset from {config.dataset_path}...")
raw_data = torch.load(config.dataset_path)
train_data = raw_data['train']
X_val, Y_val = raw_data['val']

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

model.train()
for step in range(config.max_steps):
    t0 = time.time()
    x, y = get_batch(train_data, config.batch_size, config.device)

    B, T = x.shape
    init_l, init_o = model.get_initial_state()
    
    latents = init_l.view(1, 1, -1).expand(B, T + config.num_register_tokens, -1).clone()
    outputs = init_o.view(1, 1, -1).expand(B, T + config.num_register_tokens, -1).clone()
    
    total_loss_accum = 0.0
    active_batch_mask = torch.ones(B, dtype=torch.bool, device=config.device)
    
    # deep refinement
    for recurrent_step in range(config.max_recurrent_steps):
        optimizer.zero_grad()
        
        # one forward pass
        curr_x = x[active_batch_mask]
        curr_y = y[active_batch_mask]

        curr_x = curr_x.clone()
        curr_x[curr_x == -100] = PAD_TOKEN_ID
        
        loss_val, (ce_loss, halt_loss), next_latents, next_outputs, logits, halt_probs = \
            model(curr_x, latents, outputs, targets=curr_y)
        
        # backward after every step
        loss_val.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss_accum += loss_val.item()

        should_halt = (halt_probs >= config.halt_prob_threshold)
        not_halted = ~should_halt
        
        if not not_halted.any():
            # everyone finished!
            break
            
        # filter states for next step
        latents = next_latents[not_halted]
        outputs = next_outputs[not_halted]
        
        # slice the *currently active* mask
        current_active_indices = torch.nonzero(active_batch_mask).squeeze()

        if current_active_indices.dim() == 0: current_active_indices = current_active_indices.unsqueeze(0)
            
        indices_to_keep = current_active_indices[not_halted]
        
        new_mask = torch.zeros_like(active_batch_mask)
        new_mask[indices_to_keep] = True
        active_batch_mask = new_mask

    dt = time.time() - t0
    if step % 100 == 0:
        pad_id = PAD_TOKEN_ID
        X_val_clean = X_val.clone()
        # Replace any -100s in Input with Pad ID (just in case)
        X_val_clean[X_val_clean == -100] = pad_id 

        preds, steps_taken = model.predict(
            X_val_clean, 
            halt_threshold=0.5, 
            max_deep_refinement_steps=config.max_recurrent_steps
        )
        x_row = X_val_clean[0]
        y_row = Y_val[0]
        pred_row = preds[0]
        valid_indices = (y_row != -100)

        prompt_tokens = x_row[~valid_indices]
        prompt_tokens = prompt_tokens[prompt_tokens != pad_id]
        prompt_str = decode(prompt_tokens.tolist())

        pred_tokens = pred_row[valid_indices]
        pred_str = decode(pred_tokens.tolist())
        print(f'{prompt_str} {pred_str}')

        print(f"Step {step:4d} | Batch Loss: {total_loss_accum:.4f} | "
              # f"Val Acc: {val_acc*100:.2f}% | "
              f"Time: {dt*1000:.1f}ms | Recur Steps: {recurrent_step+1}")

    if step % 500 == 0 and step > 0:
        # Simple checkpoint
        print("Saving checkpoint...")
        torch.save(model.state_dict(), f"recursive_ckpt_{step}.pt")

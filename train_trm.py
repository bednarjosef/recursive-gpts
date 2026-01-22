import torch
from torch.utils.data import Dataset, DataLoader

from custom_trm import TinyRecursiveModel
from custom_mlp_mixer import MLPMixer1D
from custom_trainer import Trainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_path = 'math_dataset.pt'
print(f"Loading dataset from {dataset_path}...")
data = torch.load(dataset_path)

vocab = data['vocab']
vocab_size = len(vocab)
block_size = data['config']['max_prob_len_limit']

itos = { i:ch for ch,i in vocab.items() }
encode = lambda s: [vocab[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

class TrainDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_dataset = TrainDataset(*data['train'])
val_dataset = TrainDataset(*data['val'])

print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")
print(f"Block Size (Seq Len): {block_size}")
print(f"Vocab Size: {vocab_size}")

num_registers = 8
model_dim = 32

mixer_seq_len = (block_size - 0) + num_registers

trm = TinyRecursiveModel(
    dim = model_dim,
    num_tokens = vocab_size,
    num_register_tokens = num_registers,
    
    network = MLPMixer1D(
        dim = model_dim,
        depth = 1,
        seq_len = mixer_seq_len
    )
)
trm.to(device)

params = sum(p.numel() for p in trm.parameters())
print(f"Model Parameters: {params:,}")


optim = torch.optim.AdamW(trm.parameters(), lr=1e-3, weight_decay=0.1)

trainer = Trainer(
    trm,
    train_dataset,
    optim=optim,
    epochs = 20,
    batch_size = 128,
    learning_rate = 3e-4,
    weight_decay = 0.1,
    max_recurrent_steps = 4,
    halt_prob_thres = 0.5,
    cpu = False if device == 'cuda' else True,
    accelerate_kwargs = dict(mixed_precision = 'no'),
    eval_interval = 100,
    decode = decode,
    val_dataset = val_dataset,
)

# ==========================================
# 4. Train
# ==========================================
print("\nStarting Training...")
trainer()

torch.save(trm.state_dict(), 'saved-trm-arc-agi.pt')
print("Model saved to saved-trm-gsm")

# ==========================================
# 5. Evaluation / Inference
# ==========================================
# print("\nRunning Evaluation on Validation Set...")

# trm.eval()
# device = next(trm.parameters()).device
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# total_correct = 0
# total_samples = 0

# # For visualization
# shown_examples = 0

# with torch.no_grad():
#     for x, y in val_loader:
#         x, y = x.to(device), y.to(device)
        
#         # Predict using dynamic halting
#         # Using a higher max step during inference to allow "thinking"
#         preds, exit_steps = trm.predict(
#             x,
#             max_deep_refinement_steps = trainer.max_recurrent_steps, 
#             halt_prob_thres = 0.5
#         )
        
#         # Calculate Accuracy
#         # Only check tokens where y != -100 (the answer part)
#         mask = (y != -100)
        
#         # Check correctness
#         # Correct if: (Pred matches Target) OR (Target is masked/prompt)
#         token_match = (preds == y) | (~mask)
#         row_match = token_match.all(dim=1)
        
#         total_correct += row_match.sum().item()
#         total_samples += x.size(0)
        
#         # Visualize a few failures and successes
#         if shown_examples < 5:
#             for i in range(x.size(0)):
#                 if shown_examples >= 5: break
                
#                 # Decode
#                 prompt_mask = (y[i] == -100)
#                 # Filter out padding (assumed to be | in your vocab)
#                 pad_id = data['meta']['stoi']['|']
                
#                 prompt_ids = x[i][prompt_mask]
#                 prompt_ids = prompt_ids[prompt_ids != pad_id]
#                 prompt_str = decode(prompt_ids.tolist())
                
#                 truth_ids = y[i][~prompt_mask]
#                 truth_str = decode(truth_ids.tolist())
                
#                 pred_ids = preds[i][~prompt_mask]
#                 pred_str = decode(pred_ids.tolist())
                
#                 status = "✅" if row_match[i] else "❌"
#                 steps = exit_steps[i].item()
                
#                 print(f"Prob: {prompt_str} | Truth: {truth_str} | Pred: {pred_str} | Steps: {steps} {status}")
#                 shown_examples += 1

# accuracy = total_correct / total_samples
# print(f"\nFinal Validation Accuracy: {accuracy*100:.2f}% was achieved with {params:,} parameters.")
# print(f'Resulting parameter efficiency (% / 1k params) is {((accuracy*100) / (params / 1000)):.2f}.')

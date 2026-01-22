import json
import torch
import random
from pathlib import Path

# ==========================================
# 1. Vocabulary & Configuration
# ==========================================

# Vocabulary Definition
# 0-9: Colors
# |: Row Separator
# ,: Color Separator
# I: Input Prefix Start
# O: Output Prefix Start
# :: Prefix End (used in 'I:' and 'O:')
# ~: End of Token (EOS)
# _: Padding

VOCAB = '0123456789|,:IO~_'
vocab_size = len(VOCAB)

stoi = {ch: i for i, ch in enumerate(VOCAB)}
itos = {i: ch for i, ch in enumerate(VOCAB)}

def encode(s):
    # Safety check to ignore chars not in vocab (like newlines)
    return [stoi[c] for c in s if c in stoi]

def decode(l):
    return ''.join([itos[i] for i in l])

PAD_TOKEN = '_'
EOS_TOKEN = '~'
PAD_ID = stoi[PAD_TOKEN]

# ==========================================
# 2. ARC Processing Logic (Adapted)
# ==========================================

example_in_prefix = 'I:'
example_out_prefix = 'O:'
color_separator = ','
row_separator = '|'

def convert_to_sequence(grid: list):
    buffer = ''
    for row in grid:
        for color in row:
            if len(buffer) > 0 and buffer[-1] not in ['|', ',', ':']:
                buffer = buffer + color_separator + str(color)
            else:
                buffer = buffer + str(color)
        buffer = buffer + row_separator
    return buffer

def process_json_to_pairs(data):
    """
    Extracts (input_str, output_str) tuples from a single JSON object.
    """
    train_pairs = []
    val_pairs = []

    # Process Train Examples
    for item in data['train']:
        # Format: I:1,2|3,4O:
        inp_str = example_in_prefix + convert_to_sequence(item['input']) + example_out_prefix
        # Format: 1,2|3,4~
        out_str = convert_to_sequence(item['output']) + EOS_TOKEN
        train_pairs.append((inp_str, out_str))

    # Process Test Examples (Validation)
    for item in data['test']:
        inp_str = example_in_prefix + convert_to_sequence(item['input']) + example_out_prefix
        out_str = convert_to_sequence(item['output']) + EOS_TOKEN
        val_pairs.append((inp_str, out_str))
        
    return train_pairs, val_pairs

# ==========================================
# 3. Load and Process Data
# ==========================================

data_dir = Path("./data/evaluation")
if not data_dir.exists():
    raise FileNotFoundError(f"Directory {data_dir} not found. Please create it and add ARC .json files.")

json_files = list(data_dir.glob("*.json"))
print(f"Found {len(json_files)} JSON files.")

all_train_raw = []
all_val_raw = []

for filepath in json_files:
    with open(filepath) as f:
        data = json.load(f)
        t, v = process_json_to_pairs(data)
        all_train_raw.extend(t)
        all_val_raw.extend(v)

print(f"Extracted {len(all_train_raw)} training examples and {len(all_val_raw)} validation examples.")

# ==========================================
# 4. Determine Block Size
# ==========================================

max_seq_len = 0
for inp, out in all_train_raw + all_val_raw:
    curr_len = len(inp) + len(out)
    if curr_len > max_seq_len:
        max_seq_len = curr_len

# Add a small buffer to block size
BLOCK_SIZE = int(max_seq_len * 1.1)
print(f"Max sequence length found: {max_seq_len}")
print(f"Setting BLOCK_SIZE to: {BLOCK_SIZE}")

# ==========================================
# 5. Tokenize and Create Tensors
# ==========================================

def create_tensor_lists(raw_pairs):
    x_list = []
    y_list = []
    
    for inp_str, out_str in raw_pairs:
        full_str = inp_str + out_str
        
        # 1. Encode
        encoded = encode(full_str)
        
        # Skip if longer than block size (shouldn't happen due to logic above, but safe check)
        if len(encoded) > BLOCK_SIZE:
            continue
            
        # 2. Pad
        padding = [PAD_ID] * (BLOCK_SIZE - len(encoded))
        encoded = encoded + padding
        
        # 3. Create X and Y (Shifted)
        x_enc = encoded[:-1]
        y_enc = encoded[1:]
        
        # 4. Masking
        # We want to ignore loss for the input prompt.
        # The input prompt ends just before the Output grid starts.
        # inp_str is "I:...O:", so its length determines the split point.
        
        split_index = len(inp_str) - 1 # -1 because y is shifted by 1 relative to full string
        
        # Apply mask: -100 for all positions belonging to input prompt
        y_enc = [-100 if i < split_index else t for i, t in enumerate(y_enc)]
        
        x_list.append(x_enc)
        y_list.append(y_enc)
        
    return x_list, y_list

print("Tokenizing training data...")
x_train, y_train = create_tensor_lists(all_train_raw)

print("Tokenizing validation data...")
x_val, y_val = create_tensor_lists(all_val_raw)

# Convert to PyTorch Tensors
X_train = torch.tensor(x_train, dtype=torch.long)
Y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(x_val, dtype=torch.long)
Y_val = torch.tensor(y_val, dtype=torch.long)

# ==========================================
# 6. Save
# ==========================================

filepath = 'arc_dataset.pt'

print(f'Train tensor shape: {X_train.shape} | Val tensor shape: {X_val.shape}')

torch.save({
    'train': (X_train, Y_train),
    'val': (X_val, Y_val),
    'meta': {
        'vocab': VOCAB,
        'stoi': stoi,
        'itos': itos,
        'block_size': BLOCK_SIZE,
        'vocab_size': vocab_size
    }
}, filepath)

print(f'Saved dataset to {filepath}')

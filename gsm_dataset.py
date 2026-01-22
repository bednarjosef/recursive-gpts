import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm
import string


DATASET_NAME = "nvidia/OpenMathInstruct-2"
SPLIT = "train_5M"
OUTPUT_FILENAME = "math_dataset.pt"

USE_COT_AS_TARGET = False 

MAX_PROBLEM_LENGTH = 1024
PAD_TOKEN_ID = 0
EOT_TOKEN_ID = 1

CHARS = sorted(list(set(string.printable)))
VOCAB = {c: i + 2 for i, c in enumerate(CHARS)}
VOCAB['<PAD>'] = PAD_TOKEN_ID
VOCAB['<EOT>'] = EOT_TOKEN_ID

def tokenize(text):
    """Converts string to list of integer token IDs."""
    if not isinstance(text, str):
        text = str(text)
    return [VOCAB.get(c, VOCAB.get('?')) for c in text]

def pad_and_tensorize(batch_list, max_len=None):
    if not batch_list:
        return torch.tensor([])
        
    if max_len is None:
        max_len = max(len(x) for x in batch_list)

    tensor_batch = torch.full((len(batch_list), max_len), PAD_TOKEN_ID, dtype=torch.long)
    
    for i, seq in enumerate(batch_list):
        if len(seq) > max_len: # Truncate if necessary (though we filter inputs)
            seq = seq[:max_len]
        tensor_batch[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        
    return tensor_batch

def main():
    print(f"Loading {DATASET_NAME} ({SPLIT})...")
    ds = load_dataset(DATASET_NAME, split=SPLIT)  # streaming=True
    filtered_data = []
    
    print("Filtering data (Source: augmented_gsm8k, Len < 1024)...")
    
    for sample in tqdm(ds):
        if sample.get('problem_source') != 'augmented_gsm8k':
            continue
            
        problem = sample['problem']
        
        x_tokens = tokenize(problem)
        if len(x_tokens) > MAX_PROBLEM_LENGTH:
            continue

        if USE_COT_AS_TARGET:
            target_text = sample['generated_solution']
            if len(target_text) > MAX_PROBLEM_LENGTH:
                continue
        else:
            target_text = sample['expected_answer']
            
        y_tokens = tokenize(target_text)
        y_tokens.append(EOT_TOKEN_ID)

        filtered_data.append((x_tokens, y_tokens))

    print(f"Total valid samples found: {len(filtered_data)}")

    if len(filtered_data) < 100:
        raise ValueError("Not enough data found for the validation split!")

    import random
    random.seed(42)
    random.shuffle(filtered_data)

    # Take 100 for validation
    val_raw = filtered_data[:100]
    train_raw = filtered_data[100:]

    print(f"Train size: {len(train_raw)}")
    print(f"Val size: {len(val_raw)}")

    print("Tensorizing and padding...")    
    # Separate X and Y
    train_x_list = [item[0] for item in train_raw]
    train_y_list = [item[1] for item in train_raw]
    
    val_x_list = [item[0] for item in val_raw]
    val_y_list = [item[1] for item in val_raw]

    # Create Tensors
    # We pad to the max length found in the specific split to save space
    train_x_tensor = pad_and_tensorize(train_x_list)
    train_y_tensor = pad_and_tensorize(train_y_list)
    
    val_x_tensor = pad_and_tensorize(val_x_list)
    val_y_tensor = pad_and_tensorize(val_y_list)

    save_dict = {
        'train': (train_x_tensor, train_y_tensor),
        'val': (val_x_tensor, val_y_tensor),
        'vocab': VOCAB,
        'config': {
            'use_cot': USE_COT_AS_TARGET,
            'max_prob_len_limit': MAX_PROBLEM_LENGTH
        }
    }

    print(f"Saving to {OUTPUT_FILENAME}...")
    torch.save(save_dict, OUTPUT_FILENAME)
    print("Done!")

if __name__ == "__main__":
    main()

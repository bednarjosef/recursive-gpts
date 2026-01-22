import torch
from torch.utils.data import Dataset, DataLoader


dataset_path = 'arc_dataset.pt'
print(f"Loading dataset from {dataset_path}...")
data = torch.load(dataset_path)

vocab = data['meta']['vocab']
vocab_size = data['meta']['vocab_size']
block_size = data['meta']['block_size']
itos = data['meta']['itos']
decode = lambda l: ''.join([itos[i] for i in l])


class AdditionDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

        self.x[self.x == -100] = 0

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_dataset = AdditionDataset(*data['train'])
val_dataset = AdditionDataset(*data['val'])

print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")
print(f"Block Size (Seq Len): {block_size}")
print(f"Vocab Size: {vocab_size}")

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


batch = 0
for dataset_input, dataset_output in train_loader:
    batch += 1
    if batch <= 3:
        for i in range(dataset_input.size(0)):
            print(dataset_input.tolist())

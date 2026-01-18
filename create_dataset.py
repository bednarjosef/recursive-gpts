import random, torch

EOT_TOKEN = '~'
PAD_TOKEN = '|'

vocab = '0123456789+=' + EOT_TOKEN + PAD_TOKEN
vocab_size = len(vocab)

stoi = { ch:i for i,ch in enumerate(vocab) }
itos = { i:ch for i,ch in enumerate(vocab) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

EOT_TOKEN_ID = stoi[EOT_TOKEN]
PAD_TOKEN_ID = stoi[PAD_TOKEN]

MAX_DIGITS = 2
BLOCK_SIZE = (3 * MAX_DIGITS) + 4
MAX_ROWS = 10_000
VAL_SPLIT = 0.02
INCLUDE_REVERSE = True
filepath = 'addition_dataset_10k_2dig.pt'

numbers = list(range(10 ** MAX_DIGITS))

pairs = []
for a in numbers:
    for b in numbers:
        if not INCLUDE_REVERSE and (b, a) in pairs:
            continue
        pairs.append((a, b))

if len(pairs) > MAX_ROWS:
    print(f'Picking {MAX_ROWS} random pairs...')
    pairs = random.sample(pairs, MAX_ROWS)

random.shuffle(pairs)

x_list = []
y_list = []

for pair in pairs:
    a, b = pair
    result = a + b
    seq = f'{a}+{b}={result}{EOT_TOKEN}'
    padding = (BLOCK_SIZE - len(seq)) * PAD_TOKEN
    seq = seq + padding

    encoded_seq = encode(seq)

    x_enc = encoded_seq[:-1]
    y_enc = encoded_seq[1:]

    # mask stuff before the = sign
    eq_index = x_enc.index(stoi['='])
    y_enc = [-100 if i < eq_index else t for i, t in enumerate(y_enc)]

    x_list.append(x_enc)
    y_list.append(y_enc)


X = torch.tensor(x_list, dtype=torch.long)
Y = torch.tensor(y_list, dtype=torch.long)

n = int((1 - VAL_SPLIT) * len(pairs))

X_train, X_val = X[:n], X[n:]
Y_train, Y_val = Y[:n], Y[n:]

print(f'Train split: {len(X_train)} | Val split: {len(X_val)}')

torch.save({
    'train': (X_train, Y_train),
    'val': (X_val, Y_val),
    'meta': {
        'vocab': vocab,
        'stoi': stoi,
        'itos': itos,
        'block_size': BLOCK_SIZE,
        'vocab_size': vocab_size
    }
}, filepath)

print(f'Saved to {filepath}')

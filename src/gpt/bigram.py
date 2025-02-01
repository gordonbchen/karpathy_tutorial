from pathlib import Path

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

BATCH_SIZE = 23
BLOCK_SIZE = 8
TRAIN_STEPS = 3_000
LR = 1e-2
EVAL_INTERVAL = 500
EVAL_ITERS = 200

# Download data.
data_path = Path("data/shakespeare.txt")
if not data_path.exists():
    text = requests.get(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    ).text
    with open(data_path, mode="w") as f:
        f.write(text)
else:
    with open(data_path, mode="r") as f:
        text = f.read()

# Tokenize.
chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)

ctoi = {c: i for (i, c) in enumerate(chars)}
itoc = {i: c for (i, c) in enumerate(chars)}


def encode(string):
    return [ctoi[c] for c in string]


def decode(inds):
    return "".join([itoc[i] for i in inds])


data = torch.tensor(encode(text), dtype=torch.long, device="cuda")
n_train = int(0.9 * len(data))
train_data = data[:n_train]
val_data = data[n_train:]


def get_batch(data):
    idxs = torch.randint(0, len(data) - BLOCK_SIZE, (BATCH_SIZE,), device="cuda")
    xb = torch.stack([data[i : i + BLOCK_SIZE] for i in idxs])
    yb = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in idxs])
    return xb, yb


# Define model.
class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, xb, yb=None):
        logits = self.embedding(xb)

        if yb is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), yb.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, xb, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(xb)
            logits = logits[:, -1, :]  # B,T,C -> B, C
            probs = F.softmax(logits, dim=-1)
            new_inds = torch.multinomial(probs, num_samples=1)  # B
            xb = torch.cat((xb, new_inds), dim=-1)  # B,T -> B,T+1
        return xb


# Create model and train.
@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_steps):
    model.eval()
    total_train_loss, total_val_loss = 0, 0
    for _ in range(eval_steps):
        _, train_loss = model(*get_batch(train_data))
        _, val_loss = model(*get_batch(val_data))

        total_train_loss += train_loss.item()
        total_val_loss += val_loss.item()

    model.train()
    return total_train_loss / eval_steps, total_val_loss / eval_steps


bigram = BigramLM(VOCAB_SIZE).to("cuda").train()
optim = AdamW(bigram.parameters(), lr=LR)

for i in range(TRAIN_STEPS):
    xb, yb = get_batch(train_data)
    _, loss = bigram(xb, yb)

    optim.zero_grad()
    loss.backward()
    optim.step()

    if (i % EVAL_INTERVAL == 0) or (i == TRAIN_STEPS - 1):
        train_loss, val_loss = estimate_loss(bigram, train_data, val_data, EVAL_ITERS)
        print(f"{i}: {train_loss=} {val_loss=}")

bigram.eval()
new_x = torch.zeros((1, 1), dtype=torch.long, device="cuda")
print(decode(bigram.generate(new_x, 500)[0].tolist()))

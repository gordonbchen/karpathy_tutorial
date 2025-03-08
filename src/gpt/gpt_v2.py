import math
from pathlib import Path

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

# Hyperparams.
BATCH_SIZE = 64
BLOCK_SIZE = 128
TRAIN_STEPS = 5_000
LR = 3e-4
EVAL_INTERVAL = 500
EVAL_ITERS = 128
EMBED_DIM = 256
N_HEADS = 8
N_LAYERS = 6
DROPOUT = 0.2

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
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        assert embed_dim % n_heads == 0, f"{embed_dim=} must be a multiple of {n_heads=}"
        self.head_size = embed_dim // n_heads
        self.qkv_linear = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.register_buffer(
            "tril", torch.triu(torch.ones(BLOCK_SIZE, BLOCK_SIZE, dtype=torch.bool), diagonal=1)
        )

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        qkv = self.qkv_linear(x)
        # Split QKV: (B,T,3*C) -> 3*(B,T,C).
        q, k, v = qkv.tensor_split(3, dim=-1)

        # Batch heads: (B,T,C) -> (B,n_heads,T,head_size).
        B, T, C = x.shape
        q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_size).transpose(1, 2)

        wei = q @ k.transpose(-1, -2)
        wei.masked_fill_(self.tril[:T, :T], float("-inf"))
        wei = F.softmax(wei / (self.head_size**0.5), dim=-1)
        out = wei @ v

        # Unbatch heads: (B,nh,T,hs) -> (B,T,C).
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, n_heads)
        self.ffwd = FeedFoward(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_embedding = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.blocks = nn.Sequential(*[Block(EMBED_DIM, N_HEADS) for i in range(N_LAYERS)])
        self.ln = nn.LayerNorm(EMBED_DIM)
        self.lm_head = nn.Linear(EMBED_DIM, VOCAB_SIZE)

        self.register_buffer("positions", torch.arange(BLOCK_SIZE))

    def forward(self, xb):
        tok_emb = self.tok_embedding(xb)  # (B,T) -> (B,T,EMBED_DIM)
        pos_emb = self.pos_embedding(self.positions[: xb.shape[-1]])  # (T,) -> (T,EMBED_DIM)
        x = tok_emb + pos_emb  # (B,T,EMBED_DIM)
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)  # (B,T,EMBED_DIM) -> (B,T,VOCAB_SIZE)
        return logits

    def calc_loss(self, xb, yb):
        logits = self(xb)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), yb.view(B * T))
        return loss

    @torch.no_grad()
    def generate(self, xb, max_new_tokens):
        for _ in range(max_new_tokens):
            ctx = xb[:, -BLOCK_SIZE:]
            logits = self(ctx)
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
        train_loss = model.calc_loss(*get_batch(train_data))
        val_loss = model.calc_loss(*get_batch(val_data))

        total_train_loss += train_loss.item()
        total_val_loss += val_loss.item()

    model.train()
    return total_train_loss / eval_steps, total_val_loss / eval_steps


gpt = torch.compile(GPT().to("cuda").train())
optim = AdamW(gpt.parameters(), lr=LR)

print(f"Params: {sum(p.numel() for p in gpt.parameters())}")
print(f"Expected init loss: {math.log(VOCAB_SIZE)}")

logger = SummaryWriter()
for i in range(TRAIN_STEPS):
    xb, yb = get_batch(train_data)
    loss = gpt.calc_loss(xb, yb)

    optim.zero_grad()
    loss.backward()
    optim.step()

    if (i % EVAL_INTERVAL == 0) or (i == TRAIN_STEPS - 1):
        train_loss, val_loss = estimate_loss(gpt, train_data, val_data, EVAL_ITERS)
        logger.add_scalars("loss", {"train": train_loss, "val": val_loss}, i)
        print(f"{i}: {train_loss=} {val_loss=}")
logger.close()

gpt.eval()
new_x = torch.zeros((1, 1), dtype=torch.long, device="cuda")
new_text = decode(gpt.generate(new_x, 1_000)[0].tolist())
print(new_text)

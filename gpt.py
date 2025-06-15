import torch
import torch.nn as nn
from torch.nn import functional as F
from torchinfo import summary
from tqdm import tqdm, trange

batch_size = 32
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(42)

with open("data/raw/input.txt", "r") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}
encode = lambda s: [char2int[ch] for ch in s]
decode = lambda x: "".join([int2char[i] for i in x])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.d = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        v = self.value(x)  # (B, T, hs)

        attn = q @ k.transpose(-2, -1) / (self.d**0.5)  # (B, T, T)
        attn.masked_fill_(self.tril[:T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        return attn @ v  # (B, T, hs)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_head: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_head)])
        self.proj = nn.Linear(num_head * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.Mish(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class GPTBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[GPTBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(-1))
            return logits, loss
        else:
            return logits, None

    def generate(self, idx, max_len=100):
        with torch.no_grad():
            for t in range(max_len):
                logits, _ = self(idx)  # (B, T, C)
                logits = logits[:, -1, :]  # (B, C)
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, idx_next], dim=1)

        return idx


model = GPTLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

summary(model, input_size=(batch_size, block_size), dtypes=[torch.long])

for iter in trange(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        tqdm.write(
            f"iter {iter:6d} train loss: {losses['train']:.2f} val loss: {losses['val']:.2f}"
        )

    xb, yb = get_batch("train")
    optimizer.zero_grad()
    logits, loss = model(xb, yb)
    loss.backward()
    optimizer.step()

context = [encode("I am")]
context = torch.tensor(context, dtype=torch.long).to(device)
generated = model.generate(context)

print(decode(generated[0].cpu().numpy()))

import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_iters = 200

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


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.embedding(idx)

        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(-1))
            return logits, loss
        else:
            return logits, None

    def generate(self, idx, max_len=100):
        with torch.no_grad():
            for t in range(max_len):
                logits, _ = self(idx) # (B, T, C)
                logits = logits[:, -1, :] # (B, C)
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, idx_next], dim=1)

        return idx


model = BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
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

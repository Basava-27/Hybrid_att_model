import torch
import torch.nn as nn
from utils.hybrid_attention import HybridAttention
from .feedforward import GatedFeedForward

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = HybridAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = GatedFeedForward(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class HybridGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=256, n_layer=4, n_head=4, block_size=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.embed(idx) + self.pos(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is None:
            return logits, None

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        return logits, loss

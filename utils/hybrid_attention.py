import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1, local_window=8):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.local_window = local_window

        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(n_embd, n_embd)
        self.v_proj = nn.Linear(n_embd, n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (C ** 0.5)

        # Local attention mask
        mask = torch.ones(T, T, device=x.device).tril(self.local_window)
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        y = torch.matmul(attn, v)
        return self.out_proj(y)

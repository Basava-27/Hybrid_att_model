import torch.nn as nn
import torch.nn.functional as F

class GatedFeedForward(nn.Module):
    def __init__(self, n_embd, expansion=4, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, expansion * n_embd)
        self.fc2 = nn.Linear(expansion * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))

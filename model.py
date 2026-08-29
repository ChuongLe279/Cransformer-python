from typing import Any

import torch
import torch.nn as nn
import math

"""
nn.Module is a container that hold neural network's layer.
"""
class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None: 
        super().__init__()
        self.d_model = d_model
        self.vocal_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_length: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(p = dropout)
        # Create a matrix size (seq_length, d_model)
        pe = torch.zeros(seq_length, d_model)
        # Vector shape (seq_len,)
        position = torch.arange(seq_length, dtype=torch.float).unsqueeze(1)
        # Vector shape (d_model,)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Applying pe on even indices
        pe[:, ::2] = torch.sin(position * div_term)
        # Applying pe on odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # type: ignore # (batch, seq_len, d_model) 
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 10** - 6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        std = x.std(dims = -1, keepdim = True) # (batch, seq_len, 1)
        mean = x.mean(dims = -1, keepdim = True) # (batch, seq_len, 1)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

if __name__ == "__main__":
    model = PositionalEncoding(d_model=10, seq_length=10, dropout=0.0)
    print(model.pe)
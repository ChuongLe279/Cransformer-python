from typing import Any

import torch
import torch.nn as nn
import math


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) # (batch_size, seq_len, d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        even_idx = torch.arange(0, d_model, 2, dtype=torch.float)
        div_term = 10000 ** (2 * even_idx / d_model)
        pe[:, ::2] = torch.sin(pos / div_term)
        pe[:, 1::2] = torch.cos(pos / div_term)

        """Adding an dimension to the front"""
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, seq_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :] # type: ignore (batch_size, seq_len, d_model)
        return self.dropout(x)
    
if __name__ == "__main__":
    d_model = 2
    vocab_size = 3
    seq_len = 10
    """Contain id for the lookup table"""
    tensor = torch.tensor([ 
        [1,2,1], # 'I like cats'
        [2,1,1]  # 'You love dogs'
    ])
    input_emb = InputEmbedding(d_model, vocab_size)
    tensor = input_emb(tensor)
    print(f"Input embedding: {tensor.shape} | (batch_size, seq_len, d_model)")
    pos_encoding = PositionalEncoding(d_model=d_model, seq_len=seq_len, dropout=0.1)
    embedding = pos_encoding(tensor)
    print(f"Positional encoding: {embedding.shape} | (batch_size, seq_len, d_model)")




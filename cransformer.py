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

class AddNorm(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer)) # (batch_size, seq_len, d_model)

class FeedForward(nn.Module):
    """
    Input:  # (batch_size, seq_len, d_model)
    First Linear: # (batch_size, seq_len, 2048)
    ReLu: # (batch_size, seq_len, 2048)
    Second Linear: # (batch_size, seq_len, d_model)
    """
    def __init__(self, d_model: int, dff: int = 2048, dropout: float = 0.1) -> None:        
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(d_model, dff)     
        self.linear_2 = nn.Linear(dff, d_model)   

    def forward(self, x):
        return self.linear_2(self.dropout(self.linear_1(x)))

class Linear(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.linear(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, head: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.head = head
        self.dropout = nn.Dropout(dropout)

        assert self.d_model % self.head == 0, "d_model must be divisible by head"

        self.d_k = d_model // head
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        @staticmethod
        def attention(query, key, value):
            attention = torch.softmax(query * key.T / torch.sqrt(self.d_k)) * value
            return attention

    def forward(self, q, k, v):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)





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




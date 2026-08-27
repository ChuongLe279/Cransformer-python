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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Applying pe on even indices
        pe[:, ::2] = torch.sin(position * div_term)
        # Applying pe on odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)
        self.position = position
        self.div_term = div_term

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) #type: ignore # (batch, seq_len, d_model) 
        return self.dropout(x)
    
if __name__ == "__main__":
    model = PositionalEncoding(
        d_model=4,
        seq_length=3,
        dropout=0.0
    )

    angles = model.position * model.div_term

    print("Position:")
    print(model.position)

    print("\nDiv term:")
    print(model.div_term)

    print("\nPosition × div_term:")
    print(angles)
    print("Shape:", angles.shape)
    d_model = 4
    seq_length = 3
    dropout = 0.0
    positional_encoding = PositionalEncoding(
        d_model=d_model,
        seq_length=seq_length,
        dropout=dropout
    )

    # Fake embeddings:
    # batch_size=1, seq_length=3, d_model=4
    x = torch.zeros(1, seq_length, d_model)

    output = positional_encoding(x)

    print("\nFinal output:")
    print(output)

    print("\nFinal output shape:")
    print(output.shape)
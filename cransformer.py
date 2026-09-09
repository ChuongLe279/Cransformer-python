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
        return self.embedding(x) * math.sqrt(self.d_model)

if __name__ == "__main__":
    d_model = 512
    vocab_size = 10
    embedding = InputEmbedding(d_model, vocab_size)
    """Contain id for the lookup table"""
    tensor = torch.tensor([ 
        [1,2,3], # 'I like cats'
        [4,5,9] # 'You love dogs'
    ])
    print(f"Input embedding forward: {embedding(tensor).shape}")

import torch
import torch.nn as nn


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads)
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)

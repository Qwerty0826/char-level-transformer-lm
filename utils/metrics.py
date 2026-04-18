import torch


def perplexity(loss):
    return torch.exp(loss)

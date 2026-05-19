"""
Custom neural-network primitives built without torch.nn.functional.

All modules subclass torch.nn.Module and follow the init conventions from
the assignment:  Linear weights ~ TruncNormal(0, 2/(d_in+d_out)), clamped
at 3sigma.  Embeddings ~ TruncNormal(0,1) clamped at 3.  RMSNorm scale=1.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    """
    Bias-free linear layer: y = x W^T.

    Stores W of shape (out_features, in_features) for row-major memory
    compatibility with PyTorch.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x.to(self.W.dtype), self.W, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    """
    Embedding lookup: maps integer token IDs to dense vectors.

    Parameter shape is (num_embeddings, embedding_dim), matching the
    layout of nn.Embedding.  The weight is exposed so the LM head can
    optionally share it (weight tying).
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalisation (Zhang & Sennrich, 2019).

        RMSNorm(a_i) = (a_i / RMS(a)) * g_i
        RMS(a) = sqrt((1/d) * sum_i a_i^2 + eps)

    The forward pass upcasts inputs to float32 for numerical stability
    and returns the result in the original input dtype.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        # rsqrt(mean(x^2) + eps) is equivalent to 1/RMS
        scale = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = x_f32 * scale * self.weight.to(torch.float32)
        return out.to(in_dtype)

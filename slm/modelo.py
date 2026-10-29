"""Transformer shared by both modes, causal or bidirectional attention (box 5).

Pre-LayerNorm blocks, learned positions, GELU MLP of width 4d, dropout on the embedding, the
attention weights and both residual branches, embedding tied to the output
layer, no time input. The [MASK] logit is -inf in both modes (SPEC section 5), so the model
never predicts the mask token. Non-embedding parameters N = 12 n_layer d^2 up to biases.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm.tokenizador import ESPECIAIS

MASK = ESPECIAIS["[MASK]"]


@dataclass
class ConfigRede:
    vocab_size: int
    L: int
    n_layer: int = 6
    d: int = 384
    n_head: int = 6
    dropout: float = 0.0


class Bloco(nn.Module):
    def __init__(self, c: ConfigRede):
        super().__init__()
        self.n_head = c.n_head
        self.ln1 = nn.LayerNorm(c.d)
        self.qkv = nn.Linear(c.d, 3 * c.d)
        self.proj = nn.Linear(c.d, c.d)
        self.ln2 = nn.LayerNorm(c.d)
        self.fc = nn.Linear(c.d, 4 * c.d)
        self.out = nn.Linear(4 * c.d, c.d)
        self.dropout = c.dropout
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x: torch.Tensor, causal: bool) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.n_head, D // self.n_head).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=causal,
                                           dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.out(F.gelu(self.fc(self.ln2(x)))))


class Transformer(nn.Module):
    def __init__(self, c: ConfigRede):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab_size, c.d)
        self.pos = nn.Parameter(torch.zeros(c.L, c.d))
        self.blocos = nn.ModuleList(Bloco(c) for _ in range(c.n_layer))
        self.ln = nn.LayerNorm(c.d)
        self.drop = nn.Dropout(c.dropout)
        self.apply(self._iniciar)
        nn.init.normal_(self.pos, std=0.02)
        for b in self.blocos:
            for m in (b.proj, b.out):
                nn.init.normal_(m.weight, std=0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _iniciar(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def estados(self, ids: torch.Tensor, causal: bool) -> torch.Tensor:
        """Normalized states of the last layer, (B, T, d), the input of the output projection."""
        x = self.drop(self.emb(ids) + self.pos[: ids.shape[1]])
        for b in self.blocos:
            x = b(x, causal)
        return self.ln(x)

    def forward(self, ids: torch.Tensor, causal: bool) -> torch.Tensor:
        logits = self.estados(ids, causal) @ self.emb.weight.T
        return logits.index_fill(-1, torch.tensor([MASK], device=logits.device), float("-inf"))

    def n_parametros(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        emb = self.emb.weight.numel() + self.pos.numel()
        return {"total": total, "sem_embedding": total - emb}

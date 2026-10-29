"""Stratified masking by number of masks l (box 4).

SPEC section 5: with U ~ U(0,1) shared by the batch and a random permutation pi of the batch,
u_b = (pi(b) - 1 + U) / B and l_b = 1 + floor(u_b * L). Exactly l_b positions are masked,
uniform among the C(L, l_b) subsets.
"""
from __future__ import annotations

import torch

from slm.tokenizador import ESPECIAIS

MASK = ESPECIAIS["[MASK]"]


def sortear_l(B: int, L: int, gerador: torch.Generator | None = None) -> torch.Tensor:
    U = torch.rand((), generator=gerador)
    pi = torch.randperm(B, generator=gerador)
    u = (pi.to(torch.float64) + U.double()) / B
    return (1 + torch.floor(u * L)).long().clamp_(1, L)


def mascara_de_l(l: torch.Tensor, L: int, gerador: torch.Generator | None = None) -> torch.Tensor:
    """Boolean mask (B, L) with exactly l_b True entries per row, uniform among subsets."""
    pontos = torch.rand((l.shape[0], L), generator=gerador)
    posto = pontos.argsort(dim=1).argsort(dim=1)
    return posto < l.to(posto.device)[:, None]


def mascarar(w: torch.Tensor, mascara: torch.Tensor, mask_id: int = MASK) -> torch.Tensor:
    return w.masked_fill(mascara, mask_id)

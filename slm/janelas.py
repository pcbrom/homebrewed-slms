"""Fixed-length windows per document with EOS and predicted padding (box 3).

SPEC section 3: s(d) = tokens + EOS, cut into ceil((n+1)/L) windows of L tokens aligned to the
start, the last one padded with EOS. Padding is predicted in both modes and carries zero bytes.
No context crosses windows and documents are never concatenated.
"""
from __future__ import annotations

import math

import torch

from slm.tokenizador import ESPECIAIS

EOS = ESPECIAIS["[EOS]"]


def n_janelas(n_tokens: int, L: int) -> int:
    return math.ceil((n_tokens + 1) / L)


def janelas_documento(ids: list[int], L: int, eos: int = EOS) -> torch.Tensor:
    """Windows of one document, shape (N_w, L)."""
    nw = n_janelas(len(ids), L)
    seq = torch.full((nw * L,), eos, dtype=torch.long)
    seq[: len(ids)] = torch.as_tensor(ids, dtype=torch.long)
    return seq.view(nw, L)


def janelar(docs_ids: list[list[int]], L: int, eos: int = EOS) -> tuple[torch.Tensor, torch.Tensor]:
    """All windows of a corpus and the index of the document each window belongs to."""
    blocos = [janelas_documento(ids, L, eos) for ids in docs_ids]
    doc = torch.repeat_interleave(torch.arange(len(blocos)), torch.tensor([b.shape[0] for b in blocos]))
    return torch.cat(blocos), doc


def fracao_preenchimento(docs_ids: list[list[int]], L: int) -> float:
    """Share of window positions filled with padding EOS (SPEC section 3)."""
    total = sum(L * n_janelas(len(ids), L) for ids in docs_ids)
    uteis = sum(len(ids) + 1 for ids in docs_ids)
    return (total - uteis) / total

"""Autoregressive loss and diffusion NELBO in the number-of-masks form (box 6).

Both functions take a model with signature modelo(ids, causal) -> logits, compute the
cross-entropy in fp32 from logits.float() and return (training loss, nats per window).
SPEC sections 4 and 5. Normalising the diffusion loss by the number of masked tokens is
forbidden: the per-sequence weight is 1/l.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from slm.mascaramento import mascarar
from slm.tokenizador import ESPECIAIS

BOS = ESPECIAIS["[BOS]"]


def _ce(logits: torch.Tensor, alvo: torch.Tensor) -> torch.Tensor:
    lf = logits if logits.dtype == torch.float64 else logits.float()  # at least fp32
    return F.cross_entropy(lf.reshape(-1, lf.shape[-1]), alvo.reshape(-1), reduction="none").view(alvo.shape)


def entrada_ar(w: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.full_like(w[:, :1], BOS), w[:, :-1]], dim=1)


def perda_ar(modelo, w: torch.Tensor, devolver_logits: bool = False):
    """Loss (1/(B L)) sum_b l_AR(w_b) and l_AR per window."""
    logits = modelo(entrada_ar(w), True)
    ce = _ce(logits, w)
    nats = ce.sum(dim=1)
    perda = nats.sum() / w.numel()
    return (perda, nats, logits) if devolver_logits else (perda, nats)


def perda_difusao(modelo, w: torch.Tensor, mascara: torch.Tensor, devolver_logits: bool = False):
    """Loss (1/B) sum_b (1/l_b) sum_{i in S_b} CE and the unbiased window estimate (L/l) sum CE."""
    L = w.shape[1]
    logits = modelo(mascarar(w, mascara), False)
    ce = (_ce(logits, w) * mascara).sum(dim=1)
    l = mascara.sum(dim=1).to(ce.dtype)
    nats = ce * (L / l)
    perda = (ce / l).mean()
    return (perda, nats, logits) if devolver_logits else (perda, nats)

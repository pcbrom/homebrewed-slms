"""Ancestral sampling with float64 draws (box 7).

SPEC section 7. Logits leave the model in fp32 outside autocast and are cast to float64 for a
Gumbel-max draw, so low precision cannot truncate the noise. Temperature 1 unless a run
asks for another one (the entropy against generative perplexity diagnostic of SPEC section 14).
"""
from __future__ import annotations

import torch

from slm.mascaramento import MASK
from slm.perda import entrada_ar
from slm.tokenizador import ESPECIAIS

BOS, EOS = ESPECIAIS["[BOS]"], ESPECIAIS["[EOS]"]


def sortear(logits: torch.Tensor, gerador: torch.Generator) -> torch.Tensor:
    """Gumbel-max draw in float64 over the last dimension."""
    lp = torch.log_softmax(logits.double(), dim=-1)
    u = torch.rand(lp.shape, dtype=torch.float64, generator=gerador).to(lp.device)
    g = -torch.log(-torch.log(u.clamp_min(torch.finfo(torch.float64).tiny)))
    return (lp + g).argmax(dim=-1)


@torch.no_grad()
def gerar_difusao(modelo, L: int, T_gen: int, n: int = 1, prefixo: list[int] | None = None,
                  semente: int = 0, dispositivo: str = "cpu", temperatura: float = 1.0) -> torch.Tensor:
    """Reveal masked positions in T_gen steps; from t = k/T to s = (k-1)/T each masked position
    is revealed with probability (t - s) / t, and every position is revealed at the last step."""
    g = torch.Generator().manual_seed(semente)
    z = torch.full((n, L), MASK, dtype=torch.long)
    if prefixo:
        z[:, : len(prefixo)] = torch.tensor(prefixo)
    for k in range(T_gen, 0, -1):
        mascarado = z == MASK
        if not mascarado.any():
            break
        p_rev = 1.0 / k
        revela = mascarado & (torch.rand(z.shape, generator=g) < p_rev)
        if revela.any():
            amostra = sortear(modelo(z.to(dispositivo), False).float().cpu() / temperatura, g)
            z = torch.where(revela, amostra, z)
    return z


@torch.no_grad()
def gerar_ar(modelo, L: int, n: int = 1, prefixo: list[int] | None = None, semente: int = 0,
             dispositivo: str = "cpu", temperatura: float = 1.0) -> torch.Tensor:
    """Token-by-token draw from BOS or the prefix up to L tokens."""
    g = torch.Generator().manual_seed(semente)
    w = torch.full((n, L), EOS, dtype=torch.long)
    inicio = len(prefixo) if prefixo else 0
    if prefixo:
        w[:, :inicio] = torch.tensor(prefixo)
    for i in range(inicio, L):
        logits = modelo(entrada_ar(w)[:, : i + 1].to(dispositivo), True)[:, i].float().cpu()
        w[:, i] = sortear(logits / temperatura, g)
    return w


def ate_eos(ids: list[int]) -> list[int]:
    return ids[: ids.index(EOS)] if EOS in ids else ids

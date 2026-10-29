"""Direct preference optimisation in both generation modes (stage 3, docs/decisao_preferencia.md).

A pair is a prefix x with a preferred continuation y_w and a rejected one y_l. Each (x, y) goes into
one window of L tokens, x followed by y and EOS padding, with a boolean mask marking the positions of
y. The DPO loss (Rafailov et al., 2023) is

    -log sigmoid(beta * [(log pi(y_w|x) - log ref(y_w|x)) - (log pi(y_l|x) - log ref(y_l|x))]).

Autoregressive mode uses the exact log-probability of y given x. Diffusion mode replaces it with the
ELBO estimate restricted to the positions of y, in the number-of-masks form: l ~ U{1..|y|},
stratified over the n draws of each sequence, a uniform subset of l positions of y masked, and
(|y| / l) times the summed log-probabilities of the masked tokens, averaged over the n draws. The same
masks serve the policy and the reference (antithetic sampling, Zhu et al., 2025, section 3.2), so
the estimated margin is exactly zero when the policy equals the reference. Training then optimises
this ELBO-based surrogate, not DPO with the true diffusion likelihood.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from slm.mascaramento import mascarar
from slm.perda import _ce, entrada_ar
from slm.tokenizador import ESPECIAIS

EOS = ESPECIAIS["[EOS]"]


def montar_janelas(prefixos: list[list[int]], continuacoes: list[list[int]], L: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Windows (B, L) with x then y then EOS padding, and the mask of y positions.

    When x + y does not fit, the prefix is cut from the left; y must fit in L.
    """
    w = torch.full((len(prefixos), L), EOS, dtype=torch.long)
    m_y = torch.zeros((len(prefixos), L), dtype=torch.bool)
    for b, (x, y) in enumerate(zip(prefixos, continuacoes)):
        if len(y) > L:
            raise ValueError(f"continuacao de {len(y)} tokens nao cabe em L = {L}")
        x = x[max(0, len(x) + len(y) - L):]
        w[b, : len(x) + len(y)] = torch.tensor(x + y, dtype=torch.long)
        m_y[b, len(x): len(x) + len(y)] = True
    return w, m_y


def logp_ar(modelo, w: torch.Tensor, m_y: torch.Tensor) -> torch.Tensor:
    """Exact log p(y | x) per window, summing the log-probabilities of the y positions."""
    lp = -_ce(modelo(entrada_ar(w), True), w)
    return (lp * m_y).sum(dim=1)


def mascaras_y(m_y: torch.Tensor, n: int, gerador: torch.Generator | None = None) -> torch.Tensor:
    """n masks (n, B, L) per window, each a uniform subset of the y positions with stratified size.

    For window b with m = |y_b| positions, draw j takes l = 1 + floor(m (pi(j) - 1 + U) / n), with U
    and the permutation pi drawn per window, so the n sizes cover the n strata of [1, m].
    """
    B, L = m_y.shape
    m = m_y.sum(dim=1).to(torch.float64)
    U = torch.rand((B,), generator=gerador, dtype=torch.float64)
    pi = torch.stack([torch.randperm(n, generator=gerador) for _ in range(B)], dim=1).to(torch.float64)
    l = (1 + torch.floor(m * (pi + U) / n)).clamp(min=torch.ones_like(m), max=m).long()
    pontos = torch.rand((n, B, L), generator=gerador)
    pontos = pontos.masked_fill(~m_y.to(pontos.device), float("inf"))
    posto = pontos.argsort(dim=2).argsort(dim=2)
    return (posto < l[:, :, None]) & m_y


def elbo_difusao(modelo, w: torch.Tensor, m_y: torch.Tensor, mascaras: torch.Tensor) -> torch.Tensor:
    """ELBO estimate of log p(y | x) per window, averaged over the given masks (n, B, L)."""
    tam = m_y.sum(dim=1).to(torch.float64)
    total = torch.zeros(w.shape[0], dtype=torch.float64, device=w.device)
    for masc in mascaras:
        ce = (_ce(modelo(mascarar(w, masc), False), w) * masc).sum(dim=1).to(torch.float64)
        total = total - (tam / masc.sum(dim=1).to(torch.float64)) * ce
    return total / mascaras.shape[0]


def perda_dpo(politica, referencia, w_g, my_g, w_p, my_p, modo: str, beta: float, n: int = 8,
              gerador: torch.Generator | None = None):
    """DPO loss over a batch of pairs (g = preferred, p = rejected).

    Returns (loss, margin s per pair, preference accuracy), with s = beta * difference of log-ratios.
    The reference runs without gradient; in diffusion mode the masks are drawn once per window and
    shared by policy and reference.
    """
    if modo == "autoregressivo":
        def lp(modelo, w, my, _):
            return logp_ar(modelo, w, my)
        mg = mp = None
    else:
        def lp(modelo, w, my, masc):
            return elbo_difusao(modelo, w, my, masc)
        mg, mp = mascaras_y(my_g, n, gerador).to(w_g.device), mascaras_y(my_p, n, gerador).to(w_p.device)
    with torch.no_grad():
        ref_g, ref_p = lp(referencia, w_g, my_g, mg), lp(referencia, w_p, my_p, mp)
    s = beta * ((lp(politica, w_g, my_g, mg) - ref_g) - (lp(politica, w_p, my_p, mp) - ref_p))
    return -F.logsigmoid(s).mean(), s.detach(), (s.detach() > 0).double().mean().item()

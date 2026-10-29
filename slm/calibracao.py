"""Calibration and selective-decision criteria for next-token predictions.

Definitions follow the typed-decision class material (pcbrom/laya-jev notebook): temperature fitted
by Newton on the validation negative log-likelihood, ECE with 15 equal-width bins, reliability
diagram with over- and under-confidence regions, coverage against selective accuracy with type I
errors (wrong answer accepted) and type II errors (right answer rejected), ROC AUC of confidence
as a detector of correct predictions, and the Beta gate (point estimate against the lower 5%
quantile of Beta(hits + 1, errors + 1)). Confidence is the largest softmax probability.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.special import betaincinv
from scipy.stats import rankdata

from slm.mascaramento import mascara_de_l
from slm.perda import entrada_ar


@torch.no_grad()
def coletar_previsoes(modelo, janelas: torch.Tensor, modo: str, n_max: int = 20000, lote: int = 32,
                      semente: int = 0, dispositivo: str = "cpu", sem_preenchimento: bool = False,
                      eos: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits (fp32, CPU) and targets for up to n_max predicted positions of the given windows.

    Autoregressive: every position given its prefix. Diffusion: the masked positions of a draw of
    l ~ U{1..L} per window. With sem_preenchimento, the padding after the document [EOS] (an [EOS]
    target preceded by [EOS]) is left out: it is predicted almost surely and would dominate the
    high-confidence predictions; the [EOS] that closes the document stays.
    """
    g = torch.Generator().manual_seed(semente)
    Zs, ys, n = [], [], 0
    for i in range(0, janelas.shape[0], lote):
        if n >= n_max:
            break
        w = janelas[i:i + lote].to(dispositivo)
        if modo == "autoregressivo":
            z = modelo(entrada_ar(w), True).float()
            sel = torch.ones_like(w, dtype=torch.bool)
        else:
            L = w.shape[1]
            l = torch.randint(1, L + 1, (w.shape[0],), generator=g)
            sel = mascara_de_l(l, L, g).to(dispositivo)
            z = modelo(w.masked_fill(sel, 3), False).float()
        if sem_preenchimento:
            pad = torch.zeros_like(sel)
            pad[:, 1:] = (w[:, 1:] == eos) & (w[:, :-1] == eos)
            sel = sel & ~pad
        Zs.append(z[sel].cpu())
        ys.append(w[sel].cpu())
        n += int(sel.sum())
    return torch.cat(Zs)[:n_max], torch.cat(ys)[:n_max]


def temperatura_newton(Z: torch.Tensor, y: torch.Tensor, iteracoes: int = 100) -> float:
    """Temperature T minimising the mean NLL of softmax(Z / T), by Newton on beta = 1 / T.

    The NLL is convex in beta: its second derivative is the mean variance of Z under softmax(beta Z).
    """
    Z = Z.double()
    zy = Z.gather(1, y.view(-1, 1)).squeeze(1)
    finito = torch.isfinite(Z)
    Zf = torch.where(finito, Z, torch.zeros_like(Z))
    beta = 1.0
    for _ in range(iteracoes):
        P = torch.softmax(beta * Z, dim=1)
        Ez = (P * Zf).sum(1)
        g = float((Ez - zy).mean())
        h = float(((P * Zf ** 2).sum(1) - Ez ** 2).mean())
        passo = g / max(h, 1e-12)
        while beta - passo <= 0:
            passo /= 2
        beta -= passo
        if abs(passo) < 1e-10:
            break
    return 1.0 / beta


def confianca_acerto(Z: torch.Tensor, y: torch.Tensor, T: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    P = torch.softmax(Z.double() / T, dim=1)
    conf, pred = P.max(1)
    return conf.numpy(), (pred == y).numpy()


def ece(conf: np.ndarray, acerto: np.ndarray, faixas: int = 15) -> float:
    bordas = np.linspace(0, 1, faixas + 1)
    idx = np.clip(np.searchsorted(bordas, conf, side="left") - 1, 0, faixas - 1)
    total = 0.0
    for b in range(faixas):
        sel = idx == b
        if sel.any():
            total += sel.mean() * abs(conf[sel].mean() - acerto[sel].mean())
    return float(total)


def confiabilidade(conf: np.ndarray, acerto: np.ndarray, faixas: int = 10) -> dict:
    """Per-bin mean declared confidence, observed accuracy and count (reliability diagram)."""
    bordas = np.linspace(0, 1, faixas + 1)
    idx = np.clip(np.searchsorted(bordas, conf, side="left") - 1, 0, faixas - 1)
    conf_m, acc_m, n = [], [], []
    for b in range(faixas):
        sel = idx == b
        conf_m.append(float(conf[sel].mean()) if sel.any() else float("nan"))
        acc_m.append(float(acerto[sel].mean()) if sel.any() else float("nan"))
        n.append(int(sel.sum()))
    return {"bordas": bordas.tolist(), "confianca": conf_m, "acuracia": acc_m, "n": n}


def curva_seletiva(conf: np.ndarray, acerto: np.ndarray, pontos: int = 101) -> dict:
    """Coverage and selective accuracy as the acceptance threshold sweeps the confidence range,
    with type I errors (accepted and wrong) and type II errors (rejected and right)."""
    limiares = np.quantile(conf, np.linspace(0, 1, pontos))
    cob, acc, t1, t2 = [], [], [], []
    for tau in limiares:
        aceita = conf >= tau
        cob.append(float(aceita.mean()))
        acc.append(float(acerto[aceita].mean()) if aceita.any() else float("nan"))
        t1.append(int((aceita & ~acerto).sum()))
        t2.append(int((~aceita & acerto).sum()))
    return {"limiar": limiares.tolist(), "cobertura": cob, "acuracia": acc, "erro_tipo_I": t1, "erro_tipo_II": t2}


def auc_roc(conf: np.ndarray, acerto: np.ndarray) -> float:
    """Area under the ROC curve of confidence as a detector of correct predictions (Mann-Whitney)."""
    pos, neg = int(acerto.sum()), int((~acerto).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    r = rankdata(conf)
    return float((r[acerto].sum() - pos * (pos + 1) / 2) / (pos * neg))


def portao_beta(conf_cal, acerto_cal, conf_rel, acerto_rel, alvo: float, beta_inferior: bool = True,
                n_min: int = 20) -> dict:
    """Smallest threshold whose accepted calibration items reach `alvo`, judged by the point estimate
    or by the lower 5% quantile of Beta(hits + 1, errors + 1); then applied to the reported half."""
    tau = None
    for limiar in np.sort(np.unique(conf_cal)):
        sel = conf_cal >= limiar
        n_, a_ = int(sel.sum()), int(acerto_cal[sel].sum())
        if n_ < n_min:
            break
        est = betaincinv(a_ + 1, n_ - a_ + 1, 0.05) if beta_inferior else a_ / n_
        if est >= alvo:
            tau = float(limiar)
            break
    if tau is None:
        return {"limiar": None, "cobertura": 0.0, "acuracia_seletiva": None, "promessa_cumprida": False}
    decide = conf_rel >= tau
    acc = float(acerto_rel[decide].mean()) if decide.any() else float("nan")
    return {"limiar": tau, "cobertura": float(decide.mean()), "acuracia_seletiva": acc,
            "erro_tipo_I": int((decide & ~acerto_rel).sum()), "erro_tipo_II": int((~decide & acerto_rel).sum()),
            "promessa_cumprida": bool(acc >= alvo)}

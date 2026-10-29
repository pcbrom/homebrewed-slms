"""Typed decider on the project Transformer (docs/decisao_avaliacao.md, section 7, and
preregistro/m14_tarefas_B.json).

The encoder is an M7 checkpoint read whole, causal in the autoregressive mode and bidirectional
without masks in the diffusion mode; the mean of the last-layer states over the item tokens and
its [EOS] feeds a linear head over the options. Full fine-tuning with AdamW at the PyTorch
defaults and a constant rate, cross-entropy weighted by the inverse class frequency so that the
loss matches the balanced accuracy. Calibration by a temperature fitted on out-of-fold logits,
top-label ECE in equal-width bins, and the selective threshold as the lowest confidence whose
out-of-fold accuracy above it reaches the target.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize_scalar

from slm.janelas import EOS


class Decisor(nn.Module):
    def __init__(self, encoder: nn.Module, causal: bool, n_opcoes: int = 2):
        super().__init__()
        self.encoder, self.causal = encoder, causal
        self.cabeca = nn.Linear(encoder.c.d, n_opcoes)

    def forward(self, ids: torch.Tensor, comprimento: torch.Tensor) -> torch.Tensor:
        h = self.encoder.estados(ids, self.causal)
        m = (torch.arange(ids.shape[1], device=ids.device)[None] < comprimento[:, None]).to(h.dtype)
        return self.cabeca((h * m[..., None]).sum(1) / m.sum(1, keepdim=True))


def lote(seqs: list[list[int]], L: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Items padded with [EOS] to L, and the length of each with its own [EOS]."""
    ids = torch.full((len(seqs), L), EOS, dtype=torch.long)
    for k, s in enumerate(seqs):
        ids[k, :len(s)] = torch.as_tensor(s)
    return ids, torch.tensor([len(s) + 1 for s in seqs])


def ajustar(encoder: nn.Module, causal: bool, seqs: list[list[int]], y: np.ndarray, lr: float, L: int,
            epocas: int = 3, tam_lote: int = 16, semente: int = 0, dispositivo: str = "cpu",
            perdas: list | None = None) -> Decisor:
    torch.manual_seed(semente)
    dec = Decisor(copy.deepcopy(encoder), causal, int(y.max()) + 1).to(dispositivo).train()
    opt = torch.optim.AdamW(dec.parameters(), lr=lr)
    cont = np.bincount(y, minlength=int(y.max()) + 1)
    peso = torch.tensor(len(y) / (len(cont) * cont), dtype=torch.float32, device=dispositivo)
    rng = np.random.default_rng(semente)
    for _ in range(epocas):
        for b in np.array_split(rng.permutation(len(y)), max(1, -(-len(y) // tam_lote))):
            ids, n = lote([seqs[k] for k in b], L)
            perda = F.cross_entropy(dec(ids.to(dispositivo), n.to(dispositivo)).float(),
                                    torch.as_tensor(y[b], device=dispositivo), weight=peso)
            opt.zero_grad(set_to_none=True)
            perda.backward()
            opt.step()
            if perdas is not None:
                perdas.append(perda.item())
    return dec.eval()


@torch.no_grad()
def logits(dec: Decisor, seqs: list[list[int]], L: int, tam_lote: int = 128, dispositivo: str = "cpu") -> np.ndarray:
    saida = []
    for k in range(0, len(seqs), tam_lote):
        ids, n = lote(seqs[k:k + tam_lote], L)
        saida.append(dec(ids.to(dispositivo), n.to(dispositivo)).float().cpu().numpy())
    return np.concatenate(saida)


def temperatura(z: np.ndarray, y: np.ndarray) -> float:
    """Temperature minimizing the negative log-likelihood, searched on log T."""
    def nll(lt):
        a = z / np.exp(lt)
        return float(-(a[np.arange(len(y)), y] - np.logaddexp.reduce(a, axis=1)).mean())
    return float(np.exp(minimize_scalar(nll, bounds=(-4, 4), method="bounded").x))


def probabilidades(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    a = z / T
    return np.exp(a - np.logaddexp.reduce(a, axis=1, keepdims=True))


def ece(p: np.ndarray, y: np.ndarray, faixas: int = 15) -> float:
    """Top-label expected calibration error in equal-width confidence bins."""
    conf, acerto = p.max(1), (p.argmax(1) == y)
    idx = np.minimum((conf * faixas).astype(int), faixas - 1)
    return float(sum(abs(acerto[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
                     for b in range(faixas) if (idx == b).any()))


def limiar_seletivo(conf: np.ndarray, acerto: np.ndarray, alvo: float = 0.95) -> float:
    """Lowest confidence c whose accuracy over the items with confidence >= c reaches the target,
    infinity when none does."""
    o = np.argsort(-conf)
    ac = np.cumsum(acerto[o]) / np.arange(1, len(o) + 1)
    ok = np.flatnonzero(ac >= alvo)
    return float(conf[o][ok.max()]) if ok.size else float("inf")

"""Bits per byte and NELBO with K samples (box 8).

SPEC section 8. Evaluation runs in fp32 without autocast. For the diffusion mode each window
gets K draws of l stratified over K, with a fixed evaluation seed.
"""
from __future__ import annotations

import math

import torch

from slm.mascaramento import mascara_de_l
from slm.janelas import janelar
from slm.perda import entrada_ar, perda_ar, perda_difusao


def l_estratificado(K: int, L: int, gerador: torch.Generator) -> torch.Tensor:
    U = torch.rand((), generator=gerador).double()
    u = (torch.arange(K, dtype=torch.float64) + U) / K
    return (1 + torch.floor(u * L)).long().clamp_(1, L)


def escolher_K(C: int, N_w: int) -> int:
    if C < N_w:
        raise ValueError("evaluation budget below the number of windows: raise C, do not drop documents")
    return max(1, C // N_w)


@torch.no_grad()
def nats_por_janela(modelo, janelas: torch.Tensor, modo: str, K: int = 1, lote: int = 64,
                    semente: int = 0, dispositivo: str = "cpu") -> torch.Tensor:
    """Average nats per window, exact for the autoregressive mode, NELBO estimate for diffusion."""
    g = torch.Generator().manual_seed(semente)
    saida = []
    for i in range(0, janelas.shape[0], lote):
        w = janelas[i:i + lote].to(dispositivo)
        if modo == "autoregressivo":
            saida.append(perda_ar(modelo, w)[1].double().cpu())
            continue
        b, L = w.shape
        l = torch.cat([l_estratificado(K, L, g) for _ in range(b)])
        m = mascara_de_l(l, L, g).to(dispositivo)
        nats = perda_difusao(modelo, w.repeat_interleave(K, dim=0), m)[1].double().cpu()
        saida.append(nats.view(b, K).mean(dim=1))
    return torch.cat(saida)


def avaliar(modelo, documentos: list[dict], modo: str, L: int, K: int = 1, lote: int = 64,
            semente: int = 0, dispositivo: str = "cpu", conjunto: str = "validacao") -> dict:
    """Per-document nats and bytes and the ratio-of-sums bpb (CONTRATO_DADO section 4)."""
    janelas, doc = janelar([d["ids"] for d in documentos], L)
    nats = nats_por_janela(modelo, janelas, modo, K, lote, semente, dispositivo)
    por_doc = torch.zeros(len(documentos), dtype=torch.float64).index_add_(0, doc, nats)
    n_jan = torch.bincount(doc, minlength=len(documentos))
    linhas = [{"doc_id": d["doc_id"], "grupo": d["grupo"], "nats": float(n), "bytes": int(d["bytes"]),
               "janelas": int(j)} for d, n, j in zip(documentos, por_doc.tolist(), n_jan.tolist())]
    total_bytes = sum(r["bytes"] for r in linhas)
    return {"conjunto": conjunto, "modo": modo, "K": K if modo == "difusao" else 1, "documentos": linhas,
            "bpb": float(por_doc.sum()) / (math.log(2) * total_bytes)}


def logp_tokens_ar(modelo, docs_ids: list[list[int]], L: int, lote: int = 64, dispositivo: str = "cpu") -> list:
    """Autoregressive log-probability of every token of each document and of its [EOS], on the
    windows of the evaluation (context restarts at each window, padding after [EOS] left out).
    Used to compare and mix with the n-gram token by token."""
    janelas, doc = janelar(docs_ids, L)
    saida = []
    with torch.no_grad():
        for b in janelas.split(lote):
            b = b.to(dispositivo)
            lp = torch.log_softmax(modelo(entrada_ar(b), True).double(), -1)
            saida.append(lp.gather(-1, b.unsqueeze(-1)).squeeze(-1).cpu())
    lp = torch.cat(saida)
    por_doc, k = [], 0
    for ids in docs_ids:
        nw = int((doc == len(por_doc)).sum())
        por_doc.append(lp[k:k + nw].reshape(-1)[: len(ids) + 1].numpy())
        k += nw
    return por_doc

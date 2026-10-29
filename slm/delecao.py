"""Deletion curves for next-token decisions (adapted from the typed-decision analysis).

Item: a test passage of n context words and the next word. The decision is the token with the
largest calibrated probability at the target position given the full context, and p0 is that
probability. The importance of a word is the drop in the decided token's probability when that
word alone is deleted from the text. The deletion curve deletes words cumulatively, most
important first, and records p(decided) / p0, clipped at 1, against the fraction of words deleted
on an 11-point grid. The reference curve uses random deletion orders. At 100% the context is
empty, and the remaining probability measures the model's intrinsic bias.

Autoregressive: input BOS + context, prediction at the next position. Diffusion: input context,
the target masked and every later position of the window masked.
"""
from __future__ import annotations

import random

import numpy as np
import regex
import torch
from scipy.stats import pearsonr, spearmanr

from slm.mascaramento import MASK
from slm.tokenizador import ESPECIAIS

BOS = ESPECIAIS["[BOS]"]
GRADE = np.linspace(0, 1, 11)
_PALAVRA = regex.compile(r"\S+")


def itens(textos: list[str], tok, n_itens: int, n_palavras: int = 50, semente: int = 0) -> list[dict]:
    """Passages of n_palavras words followed by a target word, sampled from the given texts."""
    rng = random.Random(semente)
    candidatos = [(i, j) for i, t in enumerate(textos) for j in range(len(_PALAVRA.findall(t)) - n_palavras)]
    escolhidos = rng.sample(candidatos, min(n_itens, len(candidatos)))
    saida = []
    for i, j in escolhidos:
        p = _PALAVRA.findall(textos[i])
        saida.append({"palavras": p[j:j + n_palavras], "alvo": p[j + n_palavras],
                      "alvo_id": tok.encode(" " + p[j + n_palavras])[0]})
    return saida


@torch.no_grad()
def probs_proximo(modelo, modo: str, contextos: list[list[int]], L: int, T: float = 1.0, lote: int = 64,
                  dispositivo: str = "cpu") -> torch.Tensor:
    """Calibrated distribution of the next token after each context, shape (len(contextos), V)."""
    saida = []
    for s in range(0, len(contextos), lote):
        bloco = [c[-(L - 1):] for c in contextos[s:s + lote]]
        if modo == "autoregressivo":
            x = torch.full((len(bloco), L), ESPECIAIS["[EOS]"], dtype=torch.long)
            pos = []
            for b, c in enumerate(bloco):
                seq = [BOS] + c
                x[b, : len(seq)] = torch.tensor(seq)
                pos.append(len(seq) - 1)
            logits = modelo(x.to(dispositivo), True).float()
        else:
            x = torch.full((len(bloco), L), MASK, dtype=torch.long)
            pos = []
            for b, c in enumerate(bloco):
                if c:
                    x[b, : len(c)] = torch.tensor(c)
                pos.append(len(c))
            logits = modelo(x.to(dispositivo), False).float()
        sel = logits[torch.arange(len(bloco)), torch.tensor(pos, device=logits.device)]
        saida.append(torch.softmax(sel.double() / T, dim=-1).cpu())
    return torch.cat(saida)


def _ids(tok, palavras: list[str]) -> list[int]:
    return tok.encode(" ".join(palavras)) if palavras else []


def curva_item(modelo, modo: str, tok, item: dict, L: int, T: float = 1.0, sementes_aleatorias: int = 3,
               dispositivo: str = "cpu") -> dict:
    W = item["palavras"]
    n = len(W)
    p_cheio = probs_proximo(modelo, modo, [_ids(tok, W)], L, T, dispositivo=dispositivo)[0]
    d = int(p_cheio.argmax())
    p0 = float(p_cheio[d])
    sem_um = probs_proximo(modelo, modo, [_ids(tok, W[:i] + W[i + 1:]) for i in range(n)], L, T,
                           dispositivo=dispositivo)[:, d]
    importancia = (p0 - sem_um).numpy()

    def curva(ordem):
        ks = [int(round(f * n)) for f in GRADE]
        ctxs = []
        for k in ks:
            removidas = set(ordem[:k])
            ctxs.append(_ids(tok, [w for i, w in enumerate(W) if i not in removidas]))
        p = probs_proximo(modelo, modo, ctxs, L, T, dispositivo=dispositivo)[:, d].numpy()
        return np.minimum(p / p0, 1.0)

    imp = curva(list(np.argsort(-importancia, kind="stable")))
    rng = np.random.default_rng(item.get("semente", 0))
    alea = np.mean([curva(list(rng.permutation(n))) for _ in range(sementes_aleatorias)], axis=0)
    return {"decisao": d, "p0": p0, "acerto": d == item["alvo_id"], "curva_importancia": imp.tolist(),
            "curva_aleatoria": alea.tolist(), "auc_importancia": float(np.trapezoid(imp, GRADE)),
            "auc_aleatoria": float(np.trapezoid(alea, GRADE)), "p_vazio_relativo": float(imp[-1])}


def analisar(modelo, modo: str, tok, lista_itens: list[dict], L: int, T: float = 1.0,
             dispositivo: str = "cpu") -> dict:
    """Per-item curves, mean curves and the rank correlation between p0 and the deletion AUC."""
    res = [curva_item(modelo, modo, tok, it, L, T, dispositivo=dispositivo) for it in lista_itens]
    p0 = np.array([r["p0"] for r in res])
    auc = np.array([r["auc_importancia"] for r in res])
    rho = spearmanr(p0, auc)
    r = pearsonr(p0, auc) if len(res) > 2 else None
    return {"modo": modo, "T": T, "grade": GRADE.tolist(), "itens": res,
            "curva_importancia_media": np.mean([x["curva_importancia"] for x in res], axis=0).tolist(),
            "curva_aleatoria_media": np.mean([x["curva_aleatoria"] for x in res], axis=0).tolist(),
            "auc_importancia_media": float(auc.mean()),
            "auc_aleatoria_media": float(np.mean([x["auc_aleatoria"] for x in res])),
            "p_vazio_relativo_medio": float(np.mean([x["p_vazio_relativo"] for x in res])),
            "spearman_p0_auc": {"rho": float(rho.statistic), "p": float(rho.pvalue)},
            "pearson_p0_auc": {"r": float(r.statistic), "p": float(r.pvalue)} if r else None}

"""Scoring of the corpus B tasks with the project Transformer (preregistro/m14_tarefas_B.json).

Autoregressive mode: the token log-probabilities of slm.avaliacao.logp_tokens_ar feed the same
rules as the n-gram (slm.medida_tarefas.pontuar).

Diffusion mode: the conditional bound of a response span given the rest of the window, the
estimator of LLaDA (Nie et al., 2025). In each of K draws, l is stratified over 1..m, where m is
the span size, l positions of the span are masked and the draw contributes (m/l) times the summed
cross-entropy of the masked positions. The draws depend only on the item seed and on m, so options
of the same size get the same masks. With m = 1 the draw is the exact conditional of that position.
T1 scores the gap span, T2 the caput after each prefix, T3 the span where the two versions differ
(the whole window, padding included, in the shuffle family), and T4 every position after the
prefix, padding included, so the model does not learn the length of the continuation and the
estimate stays an upper bound on its nats.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from slm.avaliacao import logp_tokens_ar
from slm.janelas import janelas_documento
from slm.mascaramento import mascarar
from slm.medida_tarefas import PREFIXO_INSERCAO, PREFIXO_PROPRIO
from slm.perda import _ce


def pontuador_ar(modelo, L: int, dispositivo: str = "cpu"):
    return lambda seqs: logp_tokens_ar(modelo, seqs, L, dispositivo=dispositivo)


def semente_item(*partes) -> int:
    return int(hashlib.sha256(":".join(map(str, partes)).encode()).hexdigest()[:15], 16)


def amostras(m: int, K: int, semente: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Stratified l (K,) over 1..m and K masks over the m span positions, uniform among subsets."""
    g = torch.Generator().manual_seed(semente)
    U = torch.rand((), generator=g).double()
    l = (1 + torch.floor((torch.arange(K, dtype=torch.float64) + U) / K * m)).long().clamp_(1, m)
    posto = torch.rand((K, m), generator=g).argsort(dim=1).argsort(dim=1)
    return l, posto < l[:, None]


@torch.no_grad()
def nats_resposta(modelo, pedidos: list[tuple[list[int], int, int, int]], L: int, K: int = 128,
                  linhas: int = 2048, dispositivo: str = "cpu") -> list[float]:
    """Estimated nats of positions [ini, fim) of each single-window sequence, the rest visible.
    `pedidos` holds (ids, ini, fim, seed)."""
    saida = []
    por_lote = max(1, linhas // K)
    for c in range(0, len(pedidos), por_lote):
        ws, ms, fs = [], [], []
        for ids, ini, fim, s in pedidos[c:c + por_lote]:
            w = janelas_documento(ids, L)
            assert w.shape[0] == 1, "item longer than one window"
            l, mk = amostras(fim - ini, K, s)
            m = torch.zeros((K, L), dtype=torch.bool)
            m[:, ini:fim] = mk
            ws.append(w.expand(K, L))
            ms.append(m)
            fs.append((fim - ini) / l.double())
        w, m = torch.cat(ws).to(dispositivo), torch.cat(ms).to(dispositivo)
        ce = (_ce(modelo(mascarar(w, m), False), w) * m).sum(1).double().cpu()
        saida += (ce * torch.cat(fs)).view(-1, K).mean(1).tolist()
    return saida


def _comum(a: list[int], b: list[int]) -> tuple[int, int]:
    """Lengths of the common token prefix and suffix of two sequences."""
    p = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    s = next((k for k, (x, y) in enumerate(zip(reversed(a), reversed(b))) if x != y), min(len(a), len(b)))
    return p, min(s, min(len(a), len(b)) - p)


def pontuar_difusao(tarefa: str, itens: list[dict], modelo, codificar, L: int, K: int = 128,
                    dispositivo: str = "cpu", semente: int = 0) -> list[dict]:
    """Per-item results in the format of slm.medida_tarefas.pontuar."""
    base = lambda i: {"doc_id": i["doc_id"], "grupo": i["grupo"], "particao": i["particao"]}
    nats = lambda ped: nats_resposta(modelo, ped, L, K, dispositivo=dispositivo)
    if tarefa == "t1":
        ped, faixas = [], []
        for i in itens:
            seqs = [codificar(i["esquerda"] + o + i["direita"]) for o in i["opcoes"]]
            p = min(_comum(seqs[0], x)[0] for x in seqs[1:])
            s = min(_comum(seqs[0], x)[1] for x in seqs[1:])
            sem = semente_item(semente, i["doc_id"], i["artigo"], i["esquerda"])
            ped += [(x, p, len(x) - s, sem) for x in seqs]
        n = nats(ped)
        saida, k = [], 0
        for i in itens:
            esc = int(np.argmin(n[k:k + len(i["opcoes"])]))
            saida.append({**base(i), "familia": i["familia"], "escolha": esc, "acerto": esc == i["correta"]})
            k += len(i["opcoes"])
        return saida
    if tarefa == "t2":
        ped = []
        for i in itens:
            sem = semente_item(semente, i["doc_id"], i["texto"])
            for pref in (PREFIXO_INSERCAO, PREFIXO_PROPRIO):
                # the trailing space of the prefix merges with the first word of the caput, so the
                # response starts where the tokens of the prefix alone stop matching
                ids = codificar(pref + i["texto"])
                ped.append((ids, _comum(ids, codificar(pref))[0], len(ids), sem))
        n = nats(ped)
        return [{**base(i), "rotulo": i["rotulo"], "escore": n[2 * j + 1] - n[2 * j]} for j, i in enumerate(itens)]
    if tarefa == "t3":
        ped = []
        for i in itens:
            o, a = codificar(i["original"]), codificar(i["adulterado"])
            sem = semente_item(semente, i["doc_id"], i["original"])
            if i["familia"] == "embaralhamento":
                ped += [(o, 0, L, sem), (a, 0, L, sem)]
            else:
                p, s = _comum(o, a)
                ped += [(o, p, len(o) - s, sem), (a, p, len(a) - s, sem)]
        n = nats(ped)
        return [{**base(i), "familia": i["familia"], "acerto": bool(n[2 * j] < n[2 * j + 1])} for j, i in enumerate(itens)]
    if tarefa == "t4":
        ped = []
        for i in itens:
            ids = codificar(i["prefixo"] + i["continuacao"])
            ped.append((ids, len(codificar(i["prefixo"])), L, semente_item(semente, i["doc_id"], i["prefixo"])))
        n = nats(ped)
        return [{**base(i), "nats": x, "bytes": len(i["continuacao"].encode())} for i, x in zip(itens, n)]
    raise ValueError(tarefa)

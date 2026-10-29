"""Generative scoring of the corpus B tasks and their analysis by law group
(preregistro/m14_tarefas_B.json).

The scorer is any function that takes a list of token id sequences and returns, for each, the
log-probability of every token followed by that of [EOS]; the n-gram and the autoregressive
Transformer both fit it. In T2 the prefix terms log p(prefix) are the same for every item, so
the score is the difference of whole-sequence log-probabilities and the threshold fitted on the
training items absorbs them.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Sequence

import numpy as np
from scipy import stats

PREFIXO_INSERCAO = "passa a vigorar com a seguinte redação:\n\n“Art. 1º "
PREFIXO_PROPRIO = "Art. 1º "
LN2 = float(np.log(2.0))
Pontuador = Callable[[list[list[int]]], list[np.ndarray]]


def pontuar(tarefa: str, itens: list[dict], logp: Pontuador, codificar: Callable[[str], list[int]]) -> list[dict]:
    """Per-item result of one task: the choice in T1, the score in T2, the hit in T3 and the
    nats and bytes of the continuation in T4."""
    base = lambda i: {"doc_id": i["doc_id"], "grupo": i["grupo"], "particao": i["particao"]}
    if tarefa == "t1":
        seqs = [codificar(i["esquerda"] + o + i["direita"]) for i in itens for o in i["opcoes"]]
        tot = [float(x.sum()) for x in logp(seqs)]
        saida, k = [], 0
        for i in itens:
            n = len(i["opcoes"])
            esc = int(np.argmax(tot[k:k + n]))
            saida.append({**base(i), "familia": i["familia"], "escolha": esc, "acerto": esc == i["correta"]})
            k += n
        return saida
    if tarefa == "t2":
        lp = logp([codificar(p + i["texto"]) for i in itens for p in (PREFIXO_INSERCAO, PREFIXO_PROPRIO)])
        return [{**base(i), "rotulo": i["rotulo"], "escore": float(lp[2 * j].sum() - lp[2 * j + 1].sum())}
                for j, i in enumerate(itens)]
    if tarefa == "t3":
        lp = logp([codificar(t) for i in itens for t in (i["original"], i["adulterado"])])
        return [{**base(i), "familia": i["familia"], "acerto": bool(lp[2 * j].sum() > lp[2 * j + 1].sum())}
                for j, i in enumerate(itens)]
    if tarefa == "t4":
        seqs = [codificar(i["prefixo"] + i["continuacao"]) for i in itens]
        # the continuation starts where the tokens of the prefix alone stop matching, so a
        # tokenizer that merges the boundary space still scores the whole continuation
        n_pref = [next((k for k, (x, y) in enumerate(zip(s, codificar(i["prefixo"]))) if x != y),
                       min(len(s), len(codificar(i["prefixo"])))) for s, i in zip(seqs, itens)]
        lp = logp(seqs)
        # the last entry is [EOS], which is not part of the continuation
        return [{**base(i), "nats": float(-x[n:-1].sum()), "bytes": len(i["continuacao"].encode())}
                for i, x, n in zip(itens, lp, n_pref)]
    raise ValueError(tarefa)


def limiar_balanceado(escores: Sequence[float], rotulos: Sequence[int]) -> float:
    """Threshold on the score (inserted when above) that maximizes balanced accuracy, at a
    midpoint between consecutive distinct scores."""
    e, y = np.asarray(escores, dtype=float), np.asarray(rotulos)
    u = np.unique(e)
    cand = np.concatenate([[u[0] - 1.0], (u[:-1] + u[1:]) / 2, [u[-1] + 1.0]])
    pos, neg = e[y == 1], e[y == 0]
    ab = [((pos > c).mean() + (neg <= c).mean()) / 2 for c in cand]
    return float(cand[int(np.argmax(ab))])


def contribuicoes(tarefa: str, res: list[dict], limiar: float | None = None) -> dict[str, float]:
    """Per-group value of the task metric. T1 and T3: mean hit. T2: sum of hits weighted by
    1/(2 n_class) of the whole set, so the sum over groups is the balanced accuracy. T4: bits per
    byte of the continuations of the group."""
    por = defaultdict(list)
    for r in res:
        por[r["grupo"]].append(r)
    if tarefa in ("t1", "t3"):
        return {g: float(np.mean([r["acerto"] for r in v])) for g, v in por.items()}
    if tarefa == "t2":
        n = {c: sum(r["rotulo"] == c for r in res) for c in (0, 1)}
        # a decider gives the hit directly, a generative score needs the threshold
        acerto = lambda r: r["acerto"] if "acerto" in r else (r["escore"] > limiar) == bool(r["rotulo"])
        return {g: float(sum(acerto(r) / (2 * n[r["rotulo"]]) for r in v)) for g, v in por.items()}
    if tarefa == "t4":
        return {g: float(sum(r["nats"] for r in v) / (LN2 * sum(r["bytes"] for r in v))) for g, v in por.items()}
    raise ValueError(tarefa)


def agregar(tarefa: str, valores: Sequence[float]) -> float:
    """Main statistic from the per-group values: the sum in T2 (balanced accuracy), the macro mean otherwise."""
    return float(np.sum(valores) if tarefa == "t2" else np.mean(valores))


def residuos_media(valores: Sequence[float]) -> np.ndarray:
    """Influence of each group on the macro mean of the per-group values."""
    x = np.asarray(valores, dtype=float)
    return (x - x.mean()) / x.size


def residuos_t2(res: list[dict], grupos: Sequence[str], limiar: float | None = None) -> np.ndarray:
    """Influence of each group on the T2 balanced accuracy, by linearizing each class recall
    H_c / N_c: (h_cg - r_c n_cg) / (2 N_c) summed over the classes. A constant classifier gets
    zero influence everywhere, as its balanced accuracy is 0.5 in any sample."""
    acerto = lambda r: r["acerto"] if "acerto" in r else (r["escore"] > limiar) == bool(r["rotulo"])
    pos = {g: k for k, g in enumerate(grupos)}
    h, n = np.zeros((len(grupos), 2)), np.zeros((len(grupos), 2))
    for r in res:
        h[pos[r["grupo"]], r["rotulo"]] += acerto(r)
        n[pos[r["grupo"]], r["rotulo"]] += 1
    rec = h.sum(0) / n.sum(0)
    return ((h - rec * n) / (2 * n.sum(0))).sum(1)


def ic_selvagem(estimativa: float, residuos: Sequence[float], R: int = 10_000, semente: int = 0,
                confianca: float = 0.95) -> tuple[float, float]:
    """Wild cluster bootstrap with Rademacher weights: each replicate is the estimate plus the
    group influences with random signs. Differences between two scorers use the difference of
    their influences."""
    e = np.asarray(residuos, dtype=float)
    w = np.random.default_rng(semente).choice([-1.0, 1.0], size=(R, e.size))
    rep = estimativa + w @ e
    a = (1 - confianca) / 2
    lo, hi = np.quantile(rep, [a, 1 - a])
    return float(lo), float(hi)


def teste_sinal(d: Sequence[float]) -> tuple[int, int, float]:
    """Two-sided sign test over the groups with a nonzero difference: (positive, nonzero, p)."""
    d = np.asarray(d, dtype=float)
    nz = d[d != 0]
    pos = int((nz > 0).sum())
    return pos, int(nz.size), float(stats.binomtest(pos, nz.size, 0.5).pvalue) if nz.size else 1.0


def holm(p: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(p, dtype=float)
    o = np.argsort(p)
    aj = np.maximum.accumulate(np.minimum(1.0, (p.size - np.arange(p.size)) * p[o]))
    saida = np.empty_like(p)
    saida[o] = aj
    return saida.tolist()


def resumo_pontuador(tarefa: str, sementes: list[list[dict]], treinos: list[list[dict]] | None = None) -> dict:
    """Per-group values and influences of one scorer on one task, averaged over its seeds (one
    list of external-set results per seed). In T2 a generative scorer gets, per seed, the threshold
    fitted on its training items; a decider brings the hit and needs none."""
    grupos = sorted({r["grupo"] for r in sementes[0]})
    vals, res_g, est = [], [], []
    for k, res in enumerate(sementes):
        lim = None
        if tarefa == "t2" and "acerto" not in res[0]:
            tr = treinos[k]
            lim = limiar_balanceado([r["escore"] for r in tr], [r["rotulo"] for r in tr])
        c = contribuicoes(tarefa, res, lim)
        v = [c[g] for g in grupos]
        vals.append(v)
        res_g.append(residuos_t2(res, grupos, lim) if tarefa == "t2" else residuos_media(v))
        est.append(agregar(tarefa, v))
    return {"grupos": grupos, "valores": np.mean(vals, 0), "residuos": np.mean(res_g, 0),
            "estimativa": float(np.mean(est)), "por_semente": [float(x) for x in est]}


def comparar(tarefa: str, modelo: dict, piso: dict, R: int = 10_000) -> dict:
    """Difference model minus floor (floor minus model in T4, where lower bits per byte is
    better), its interval and the sign test over the groups."""
    assert modelo["grupos"] == piso["grupos"], "scorers measured on different groups"
    s = -1.0 if tarefa == "t4" else 1.0
    d_g = s * (modelo["valores"] - piso["valores"])
    est = s * (modelo["estimativa"] - piso["estimativa"])
    pos, nz, p = teste_sinal(d_g)
    return {"diferenca": est, "ic95": ic_selvagem(est, s * (modelo["residuos"] - piso["residuos"]), R=R),
            "grupos_a_favor": pos, "grupos_diferentes": nz, "p_sinal": p}


def icc1(y: Sequence[float], grupos: Sequence[str]) -> float:
    """Intra-group correlation ICC(1) by the one-way ANOVA estimator, for unequal group sizes."""
    y, g = np.asarray(y, dtype=float), np.asarray(grupos)
    _, idx = np.unique(g, return_inverse=True)
    n = np.bincount(idx)
    G, N = n.size, y.size
    m = np.bincount(idx, y) / n
    msb = float((n * (m - y.mean()) ** 2).sum() / (G - 1))
    msw = float(((y - m[idx]) ** 2).sum() / (N - G))
    k0 = (N - (n ** 2).sum() / N) / (G - 1)
    return (msb - msw) / (msb + (k0 - 1) * msw)

#!/usr/bin/env python3
"""Best point against best point, a post-hoc analysis declared as such (not preregistered).

Welch TOST with margin Delta_min between the best autoregressive point (64 epochs, 5 seeds) and the
best diffusion point (256 epochs, 6 seeds) of M7, on validation, where the best points were chosen,
and on test, the less biased reading. The difference is diffusion minus autoregressive, so a value
inside the margin says that the diffusion bound is not above the exact autoregressive likelihood by
more than Delta_min; it says nothing about the true likelihood of diffusion. Also the sign test over
the works (corpus A) and groups (corpus B) of the decisive comparison on test, descriptive.

    python scripts/melhor_contra_melhor.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

RAIZ = Path(__file__).resolve().parents[1]
PREFIXO = {"A": "m7-", "B": "m7-B-"}
DELTA = {"A": "m6_delta_min.json", "B": "m6_delta_min_B.json"}
ANALISES = {"A": "m7_analises.json", "B": "m7_analises_B.json"}
MELHOR = {"autoregressivo": (64, range(200, 205)), "difusao": (256, range(300, 306))}
ALFA = 0.05


def bpb(corpus, modo, arquivo):
    ep, sementes = MELHOR[modo]
    return np.array([json.loads((RAIZ / "runs" / f"{PREFIXO[corpus]}{modo}-e{ep}-s{s}" / arquivo).read_text())["bpb"]
                     for s in sementes])


def tost_welch(x, y, margem):
    """Welch TOST on mean(x) - mean(y), with the (1 - 2 alfa) interval that matches the two tests."""
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    ep = np.sqrt(vx + vy)
    gl = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    d = x.mean() - y.mean()
    p = max(stats.t.sf((d + margem) / ep, gl), stats.t.cdf((d - margem) / ep, gl))
    h = stats.t.ppf(1 - ALFA, gl) * ep
    return {"diferenca": float(d), "ic90": [float(d - h), float(d + h)], "margem": margem, "fracao_da_margem": float(d / margem),
            "gl_welch": float(gl), "p_tost": float(p), "equivalente": bool(p < ALFA)}


def main():
    saida = {"descricao": __doc__.split("\n\n")[0]}
    for c in ("A", "B"):
        margem = json.loads((RAIZ / "preregistro" / DELTA[c]).read_text())["delta_min"]
        saida[c] = {p: tost_welch(bpb(c, "difusao", arq), bpb(c, "autoregressivo", arq), margem)
                    for p, arq in (("validacao", "avaliacao.json"), ("teste", "avaliacao_teste.json"))}
        obras = json.loads((RAIZ / "resultados" / ANALISES[c]).read_text())["teste"]["por_obra"]
        neg = sum(v[0] < 0 for v in obras.values())
        saida[c]["sinal_por_unidade_teste"] = {"unidades": len(obras), "a_favor_da_difusao": neg,
                                               "p_bilateral": float(stats.binomtest(neg, len(obras), 0.5).pvalue)}
    destino = RAIZ / "resultados" / "melhor_contra_melhor.json"
    destino.write_text(json.dumps(saida, indent=1), encoding="utf-8")
    json.dump(saida, sys.stdout, indent=1)


if __name__ == "__main__":
    main()

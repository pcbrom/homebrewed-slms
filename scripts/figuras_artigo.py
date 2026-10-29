#!/usr/bin/env python3
"""Figures of the article (English labels, PDF) from resultados/ and runs/.

    python scripts/figuras_artigo.py --saida ../slm_from_scratch_artigo/figures --tabelas ../slm_from_scratch_artigo/tables

fig_epochs.pdf    validation bits per byte against epochs over the same text, both modes, every seed, 5-gram floor
fig_sentinel.pdf  per-tensor p95 relative gradient error of fp16 against fp32 in the confirmatory
                  runs, with the sentinel in training mode (dropout on) and in eval mode (dropout off)
fig_where.pdf     gain of the Transformer over the 5-gram floor per token, by context and frequency
fig_mixture.pdf   Transformer, 5-gram and their mixture on the held-out half of the validation documents
hp_search.tex     rows of the hyperparameter search, mean validation bits per byte of 2 seeds at 16 and
                  256 epochs, the configuration chosen for each mode in bold
bytes_control.tex rows of the byte-level control at equal compute against the BPE reference
fig_epochs_B.pdf  validation bits per byte against epochs on corpus B, 6-gram floor on the same windows
fig_data_curve.pdf external bits per byte against training bytes of corpus B, Transformer, floor, mixture
fig_tasks.pdf     score on the four tasks of the external set by genre (ordinary laws, decrees), with intervals and chance
fig_frontier.pdf  score on each task against training compute (6ND), with the Pareto frontier of the five models and the floor
tasks.tex         rows of the task table, estimate and interval per scorer, all acts and by genre
diagnostic.tex    rows of the surface-frequency diagnostic of T1 and T3
decider.tex       rows of the typed decider table, with the selective coverage of the project checkpoints

Every plot with a direction of improvement carries an arrow from worse to better.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
# style shared with the typed deciders audit: Latin Modern, light grid, an arrow from worse to better
plt.rcParams.update({"font.family": ["Latin Modern Roman", "DejaVu Serif"], "mathtext.fontset": "cm", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#DDDDDD",
                     "grid.linewidth": 0.6, "legend.frameon": False, "pdf.fonttype": 42})
INK, MUTED = "#14213D", "#666666"
CORES = {"autoregressivo": "#2F6FB0", "difusao": "#7A5195", "ngrama": "#C2620F", "mistura": "#5B8C5A"}
NOME = {"autoregressivo": "autoregressive", "difusao": "masked diffusion", "ngrama": "5-gram floor",
        "mistura": "mixture of both"}


def seta(ax, inicio, fim, texto="better"):
    """Arrow from worse to better in axes fraction; the label sits beside a vertical arrow and above a
    horizontal one, so it never lies on the shaft."""
    ax.annotate("", xy=fim, xytext=inicio, xycoords="axes fraction", annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.7", color=INK, lw=1.4))
    vertical = abs(fim[1] - inicio[1]) >= abs(fim[0] - inicio[0])
    meio = ((inicio[0] + fim[0]) / 2, (inicio[1] + fim[1]) / 2)
    ax.annotate(texto, xy=meio, xytext=(6, 0) if vertical else (0, 6), textcoords="offset points",
                xycoords="axes fraction", fontsize=9, color=INK, annotation_clip=False,
                ha="left" if vertical else "center", va="center" if vertical else "bottom")


def salvar(fig, caminho: Path):
    fig.savefig(caminho, bbox_inches="tight")
    plt.close(fig)


def ler(nome):
    return json.loads((RAIZ / "resultados" / nome).read_text(encoding="utf-8"))


def epocas(saida: Path):
    m7, diag = ler("m7.json"), ler("diagnostico_ngrama_A.json")
    eps = sorted(m7, key=int)
    x = [int(e) for e in eps]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for modo in ("autoregressivo", "difusao"):
        for e in eps:
            v = m7[e][f"bpb_{modo}"]
            ax.scatter([int(e)] * len(v), v, s=10, color=CORES[modo], alpha=0.45, lw=0)
        ax.plot(x, [np.mean(m7[e][f"bpb_{modo}"]) for e in eps], marker="o", ms=4, lw=1.5, color=CORES[modo],
                label=NOME[modo])
    piso = diag["bpb_ngrama_janelas"]
    ax.axhline(piso, color=CORES["ngrama"], lw=1.1, ls=(0, (4, 3)))
    ax.text(1.05, piso + 0.025, f"5-gram floor, Kneser-Ney, same windows ({piso:.3f})", fontsize=8,
            color=CORES["ngrama"])
    ax.axvline(256, color=MUTED, ls=":", lw=0.9)
    ax.set_ylim(1.72, 3.0)
    ax.text(240, 2.97, "decisive budget", fontsize=8, color=MUTED, ha="right", va="top")
    seta(ax, (0.10, 0.45), (0.10, 0.20))
    ax.set(xscale="log", xlabel="epochs over the same text (log scale)", ylabel="validation bits per byte")
    ax.set_xticks(x, eps)
    ax.minorticks_off()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9)
    salvar(fig, saida / "fig_epochs.pdf")


def sentinela(saida: Path):
    fontes = {"dropout on": RAIZ / "runs" / "_m7_sentinela_com_dropout", "dropout off": RAIZ / "runs"}
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for estilo, (rotulo, base) in zip(("--", "-"), fontes.items()):
        for modo in ("autoregressivo", "difusao"):
            v = np.sort([json.loads(l)["erro_relativo_p95"] or 0.0
                         for f in sorted(base.glob(f"m7-{modo}-e256-s30*/sentinela.jsonl"))
                         for l in f.open(encoding="utf-8")])
            ax.plot(np.maximum(v, 1e-5), np.arange(1, len(v) + 1) / len(v), ls=estilo, lw=1.5, color=CORES[modo],
                    label=f"{NOME[modo]}, {rotulo}")
    ax.axvline(0.05, color=CORES["ngrama"], lw=1.1, ls=(0, (4, 3)))
    ax.text(0.058, 0.5, "margin 0.05,\npassed on the left", fontsize=8, color=CORES["ngrama"])
    seta(ax, (0.62, 0.22), (0.40, 0.22))
    ax.set(xscale="log", xlabel="p95 relative gradient error, fp16 against fp32, per tensor",
           ylabel="cumulative fraction of records")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
    salvar(fig, saida / "fig_sentinel.pdf")


def onde(saida: Path):
    """Gain of the Transformer over the 5-gram, in nats per token, by what the n-gram knows of the
    context and by the training frequency of the token, at 16 and 64 epochs."""
    diag = ler("diagnostico_ngrama_A.json")["orcamentos"]
    paineis = (("por_contexto_ngrama", "what the 5-gram has seen",
                {"n-grama visto": "exact 5-gram\nseen in training", "contexto visto": "context seen,\nnew continuation",
                 "contexto novo": "context never\nseen"}),
               ("por_frequencia_treino", "training count of the token",
                {"0-10": "under 10", "10-100": "10 to 99", "100-1000": "100 to 999", "1000-mais": "1,000 or more"}))
    fig, eixos = plt.subplots(1, 2, figsize=(8.4, 3.6), gridspec_kw={"width_ratios": (3, 4), "wspace": 0.35})
    for ax, (chave, titulo, rotulos) in zip(eixos, paineis):
        pos = np.arange(len(rotulos))
        for k, (ep, tom) in enumerate((("16", "#8FB3DB"), ("64", CORES["autoregressivo"]))):
            g = [diag[ep][chave][r]["ganho_nats_por_token"] for r in rotulos]
            ax.bar(pos + (k - 0.5) * 0.38, g, 0.38, color=tom, label=f"autoregressive, {ep} epochs")
        frac = [diag["64"][chave][r]["fracao_tokens"] for r in rotulos]
        ax.set_xticks(pos, [f"{v}\n({100 * f:.1f}%)" for v, f in zip(rotulos.values(), frac)], fontsize=8)
        ax.set_xlabel("share of validation tokens in parentheses", fontsize=8, color=MUTED)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_title(titulo, fontsize=10, loc="left")
        ax.grid(axis="x", visible=False)
        seta(ax, (1.04, 0.35), (1.04, 0.70), "Transformer\nbetter")
    eixos[0].set_ylabel("nats per token gained over the 5-gram")
    eixos[0].legend(loc="upper center", bbox_to_anchor=(1.25, -0.32), ncol=2, fontsize=9)
    salvar(fig, saida / "fig_where.pdf")


def mistura(saida: Path):
    diag = ler("diagnostico_ngrama_A.json")["orcamentos"]
    eps = sorted(diag, key=int)
    x = [int(e) for e in eps]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for chave, nome in (("transformer", "autoregressivo"), ("ngrama", "ngrama"), ("mistura", "mistura")):
        y = [diag[e]["mistura"]["bpb_docs_impares"][chave] for e in eps]
        estilo = dict(ls=(0, (4, 3)), lw=1.1) if chave == "ngrama" else dict(marker="o", ms=4, lw=1.5)
        ax.plot(x, y, color=CORES[nome], label=NOME[nome], **estilo)
    for e in eps:
        lam = diag[e]["mistura"]["lambda_ajustado_docs_pares"]
        ax.annotate(rf"$\lambda$ = {lam:.2f}", (int(e), diag[e]["mistura"]["bpb_docs_impares"]["mistura"]),
                    xytext=(0, -15), textcoords="offset points", fontsize=8, color=CORES["mistura"],
                    ha="right" if e == eps[-1] else "center")
    ax.set_ylim(1.62, None)
    ax.text(0.98, 0.97, "mixture $p = \\lambda\\, p_{\\mathrm{Transformer}} + (1 - \\lambda)\\, p_{\\mathrm{5gram}}$\n"
            "$\\lambda$ fitted on the other half of the documents", transform=ax.transAxes, fontsize=7.5,
            color=MUTED, ha="right", va="top")
    seta(ax, (0.80, 0.70), (0.80, 0.45))
    ax.set(xscale="log", xlabel="epochs over the same text (log scale)", ylabel="bits per byte, held-out half")
    ax.set_xticks(x, eps)
    ax.minorticks_off()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=9)
    salvar(fig, saida / "fig_mixture.pdf")


def epocas_b(saida: Path):
    """Validation bits per byte against epochs on corpus B, with the 6-gram floor on the same windows
    (full training partition of the data curve)."""
    m7, piso = ler("m7_B.json"), ler("curva_dados_B.json")["fracoes"]["1.0"]["bpb_ngrama_validacao"]
    eps = sorted(m7, key=int)
    x = [int(e) for e in eps]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for modo in ("autoregressivo", "difusao"):
        for e in eps:
            v = m7[e][f"bpb_{modo}"]
            ax.scatter([int(e)] * len(v), v, s=10, color=CORES[modo], alpha=0.45, lw=0)
        ax.plot(x, [np.mean(m7[e][f"bpb_{modo}"]) for e in eps], marker="o", ms=4, lw=1.5, color=CORES[modo],
                label=NOME[modo])
    ax.axhline(piso, color=CORES["ngrama"], lw=1.1, ls=(0, (4, 3)))
    ax.text(1.05, piso + 0.03, f"6-gram floor, same windows ({piso:.3f})", fontsize=8, color=CORES["ngrama"], va="bottom")
    ax.axvline(256, color=MUTED, ls=":", lw=0.9)
    ax.text(240, 3.2, "decisive budget", fontsize=8, color=MUTED, ha="right", va="top")
    seta(ax, (0.10, 0.75), (0.10, 0.50))
    ax.set(xscale="log", xlabel="epochs over the same text (log scale)", ylabel="validation bits per byte")
    ax.set_xticks(x, eps)
    ax.minorticks_off()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9)
    salvar(fig, saida / "fig_epochs_B.pdf")


def curva(saida: Path):
    """External bits per byte against the training bytes of corpus B: autoregressive Transformer
    (every seed), 6-gram floor and their mixture."""
    fr = ler("curva_dados_B.json")["fracoes"]
    ks = sorted(fr, key=float)
    x = [fr[k]["bytes_treino"] / 1e6 for k in ks]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for chave, nome in (("bpb_transformer_externo", "autoregressivo"), ("bpb_mistura_externo", "mistura")):
        for xi, k in zip(x, ks):
            v = [s[chave] for s in fr[k]["sementes"]]
            ax.scatter([xi] * len(v), v, s=10, color=CORES[nome], alpha=0.45, lw=0)
        ax.plot(x, [np.mean([s[chave] for s in fr[k]["sementes"]]) for k in ks], marker="o", ms=4, lw=1.5,
                color=CORES[nome], label=NOME[nome])
    ax.plot(x, [fr[k]["bpb_ngrama_externo"] for k in ks], color=CORES["ngrama"], ls=(0, (4, 3)), lw=1.1,
            marker="s", ms=3, label="6-gram floor")
    seta(ax, (0.88, 0.75), (0.88, 0.50))
    ax.set(xscale="log", xlabel="training text of corpus B, MB (log scale)", ylabel="bits per byte, external laws")
    ax.set_xticks(x, [f"{v:.2f}" for v in x])
    ax.minorticks_off()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=9)
    salvar(fig, saida / "fig_data_curve.pdf")


TAREFAS = {"t1": "T1 fill the blank\n(accuracy)", "t2": "T2 inserted or own\n(balanced accuracy)",
           "t3": "T3 tampered text\n(accuracy)", "t4": "T4 continuation\n(bits per byte, reversed)"}
PONTUADORES = {"autoregressivo": ("autoregressive", CORES["autoregressivo"]),
               "difusao": ("masked diffusion", CORES["difusao"]),
               "ttl-160m": ("TeenyTinyLlama 160M", "#999999"), "tucano-160m": ("Tucano 160M", "#777777"),
               "tucano-630m": ("Tucano 630M", "#444444")}


GENEROS = {"lei": ("ordinary laws", "s"), "dec": ("decrees", "o")}
ACASO = {"t1": 0.25, "t2": 0.5, "t3": 0.5}


def tarefas(saida: Path):
    """Score of every scorer on each task of the external set by genre of the act, ordinary laws
    (filled squares) and decrees (open circles), with the 95% wild cluster bootstrap interval and
    chance as a dotted line; the T4 axis is reversed so that right is better everywhere."""
    from matplotlib.lines import Line2D
    g = ler("m14_B.json")["gerativa"]
    todos = {"ngrama": ("6-gram floor", CORES["ngrama"]), **PONTUADORES}
    fig, eixos = plt.subplots(1, 4, figsize=(9.8, 3.4), sharey=True, gridspec_kw={"wspace": 0.12})
    pos = np.arange(len(todos))[::-1]
    for ax, (t, titulo) in zip(eixos, TAREFAS.items()):
        for y, (m, (nome, cor)) in zip(pos, todos.items()):
            for dy, (gen, (_, marca)) in zip((0.17, -0.17), GENEROS.items()):
                v = g[t][m]["por_genero"][gen]
                ax.plot(v["ic95"], [y + dy] * 2, color=cor, lw=1.4)
                ax.plot(v["estimativa"], y + dy, marca, color=cor, ms=4.5, mfc=cor if gen == "lei" else "white")
        if t in ACASO:
            ax.axvline(ACASO[t], color=MUTED, lw=0.9, ls=":", zorder=1)
        ax.margins(x=0.08)
        if t == "t4":
            ax.invert_xaxis()
        ax.set_title(titulo, fontsize=9, loc="left")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=8)
        seta(ax, (0.55, -0.2), (0.95, -0.2))
    eixos[0].set_yticks(pos, [n for n, _ in todos.values()], fontsize=9)
    marcas = [Line2D([], [], marker=mk, ls="", color=INK, mfc=INK if gen == "lei" else "white", label=rot)
              for gen, (rot, mk) in GENEROS.items()] + [Line2D([], [], ls=":", color=MUTED, label="chance")]
    fig.legend(handles=marcas, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07), fontsize=9)
    fig.text(0.5, -0.12, "score on the external acts of 2025 and 2026 by genre, 95% interval", ha="center", fontsize=9, color=MUTED)
    salvar(fig, saida / "fig_tasks.pdf")


# training cost of the public yardsticks, from their papers: nominal parameters, tokens seen (TeenyTinyLlama
# Table 3, 8,192 tokens per batch over 458,000 steps, and Tucano Table 5) and training energy in kWh
# (energy section of TeenyTinyLlama and Tucano Table 12)
CUSTO_REGUAS = {"ttl-160m": (160e6, 8192 * 458_000, 15.5), "tucano-160m": (160e6, 169e9, 235.0),
                "tucano-630m": (630e6, 211e9, 920.0)}
TETO_W = 500
OURO = "#D9A21B"


def custo_casa(modo: str, ponto: int):
    """Parameters, token positions seen and an upper bound on the GPU energy (wall time at the power
    cap) of the home-built checkpoints scored in the tasks, mean over their seeds."""
    import sys
    sys.path.insert(0, str(RAIZ))
    from slm.modelo import ConfigRede, Transformer
    corridas = sorted((RAIZ / "runs").glob(f"m7-B-{modo}-e{ponto}-s*"))
    cfg = json.loads((corridas[0] / "config.json").read_text(encoding="utf-8"))
    rede = Transformer(ConfigRede(vocab_size=cfg["vocab_size"], L=cfg["L"], n_layer=cfg["n_layer"], d=cfg["d"], n_head=cfg["n_head"]))
    resumos = [json.loads((p / "avaliacao.json").read_text(encoding="utf-8"))["resumo_treino"] for p in corridas]
    tokens = np.mean([r["passos_aplicados"] for r in resumos]) * cfg["lote"] * cfg["L"]
    kwh = np.mean([r["tempo_s"] for r in resumos]) * TETO_W / 3.6e6
    return float(sum(p.numel() for p in rede.parameters())), float(tokens), float(kwh)


def fronteira(saida: Path):
    """Score on each task of the external set against training compute (6ND), with the Pareto frontier
    of the five language models and the floor; better is up and cheaper is left in every panel."""
    m14 = ler("m14_B.json")
    g = m14["gerativa"]
    custo = {m: custo_casa(m, m14["checkpoints"][m]["ponto"]) for m in ("autoregressivo", "difusao")} | CUSTO_REGUAS
    C = {m: 6 * n * d for m, (n, d, _) in custo.items()}
    curto = {"autoregressivo": "home AR", "difusao": "home diffusion", "ttl-160m": "TTL 160M",
             "tucano-160m": "Tucano 160M", "tucano-630m": "Tucano 630M"}
    titulos = {"t1": "T1 fill the blank (accuracy)", "t2": "T2 inserted or own (balanced accuracy)",
               "t3": "T3 tampered text (accuracy)", "t4": "T4 continuation (bits per byte)"}
    acaso = {"t1": 0.25, "t2": 0.5, "t3": 0.5}
    # label offsets in points, per task and model, so that no label sits on a point, bar or line
    desloc = {"t1": {"autoregressivo": (7, 8), "difusao": (7, -9)},
              "t2": {"autoregressivo": (7, -9), "difusao": (7, 10), "ttl-160m": (7, 8), "tucano-160m": (7, 14)},
              "t3": {"autoregressivo": (7, 9), "difusao": (7, -10), "ttl-160m": (7, 8), "tucano-160m": (-7, 0)},
              "t4": {"autoregressivo": (7, -9), "difusao": (7, -4), "ttl-160m": (7, -9), "tucano-160m": (7, -9),
                     "tucano-630m": (-7, 0)}}
    # the arrow sits in the empty band between the home-built models and the yardsticks
    setas = {"t1": ((0.42, 0.56), (0.27, 0.72)), "t2": ((0.40, 0.06), (0.26, 0.26)),
             "t3": ((0.36, 0.08), (0.22, 0.28)), "t4": ((0.38, 0.55), (0.24, 0.75))}
    x0, x1 = 2e15, 4e21
    fig, eixos = plt.subplots(2, 2, figsize=(7.6, 6.2), sharex=True, gridspec_kw={"hspace": 0.30, "wspace": 0.30})
    for ax, (t, titulo) in zip(eixos.flat, titulos.items()):
        inv = t == "t4"
        est = {m: g[t][m]["estimativa"] for m in PONTUADORES}
        # the floor trains in seconds on a CPU core, so it enters the frontier at the left edge
        pts = sorted([(x0, g[t]["ngrama"]["estimativa"])] + [(C[m], est[m]) for m in PONTUADORES])
        fr, melhor = [], -np.inf
        for x, y in pts:
            if (-y if inv else y) > melhor:
                fr.append((x, y))
                melhor = -y if inv else y
        xs, ys = [fr[0][0]], [fr[0][1]]
        for (_, ya), (xb, yb) in zip(fr, fr[1:]):
            xs += [xb, xb]
            ys += [ya, yb]
        ax.plot(xs + [x1], ys + [ys[-1]], color=OURO, lw=5, alpha=0.35, solid_capstyle="round", zorder=1)
        # reference labels in the empty middle band, the higher line labelled above it and the lower one below
        piso = g[t]["ngrama"]["estimativa"]
        refs = [(piso, "6-gram floor", CORES["ngrama"], (0, (4, 3)), 1.0)]
        if t in acaso:
            refs.append((acaso[t], "chance", MUTED, ":", 0.9))
        acima = max(refs, key=lambda r: -r[0] if inv else r[0])[0]
        for y, nome, cor, estilo, lw in refs:
            ax.axhline(y, color=cor, lw=lw, ls=estilo, zorder=2)
            em_cima = y == acima
            # in T2 the middle band holds the home-built points, so the labels go between the yardsticks
            ax.annotate(nome, (0.56 if t == "t2" else 0.47, y), xycoords=("axes fraction", "data"), xytext=(0, 2.5 if em_cima else -2.5),
                        textcoords="offset points", ha="left" if t == "t2" else "right", va="bottom" if em_cima else "top",
                        fontsize=7.5, color=cor)
        for m, (_, cor) in PONTUADORES.items():
            lo, hi = g[t][m]["ic95"]
            casa = m in ("autoregressivo", "difusao")
            ax.errorbar(C[m], est[m], yerr=[[est[m] - lo], [hi - est[m]]], fmt="o", color=cor, ms=6.5 if casa else 5.5,
                        mfc=cor if casa else "white", mew=1.4, lw=1.4, capsize=0, zorder=4)
            dx, dy = desloc.get(t, {}).get(m, (7, 0))
            ax.annotate(curto[m], (C[m], est[m]), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color=cor,
                        ha="left" if dx > 0 else "right", va="center", zorder=5)
        ax.set_xscale("log")
        ax.set_xlim(x0, x1)
        if inv:
            ax.invert_yaxis()
        ax.set_title(titulo, fontsize=9, loc="left")
        ax.tick_params(labelsize=8)
        ax.minorticks_off()
        seta(ax, *setas[t], "better\nand cheaper")
    for ax in eixos[1]:
        ax.set_xlabel("training compute, $6ND$ (FLOPs)", fontsize=9)
    base = C["autoregressivo"]

    def energia(m):
        return f"$\\leq${custo[m][2]:.3f} kWh" if m in ("autoregressivo", "difusao") else f"{custo[m][2]:,.1f} kWh".replace(".0 kWh", " kWh")

    longo = {"autoregressivo": "home-built autoregressive", "difusao": "home-built masked diffusion",
             "ttl-160m": "TeenyTinyLlama 160M", "tucano-160m": "Tucano 160M", "tucano-630m": "Tucano 630M"}
    alcas = [plt.Line2D([], [], marker="o", ls="none", color=cor, mfc=cor if m in ("autoregressivo", "difusao") else "white",
                        mew=1.4, ms=6) for m, (_, cor) in PONTUADORES.items()]
    # two significant figures: the token counts come from different tokenizers, so the ratio is an order of magnitude
    rotulos = [f"{longo[m]}, {custo[m][0] / 1e6:.3g}M parameters, {float(f'{C[m] / base:.2g}'):,.0f}$\\times$ compute, {energia(m)}"
               for m in PONTUADORES]
    alcas += [plt.Line2D([], [], color=OURO, lw=5, alpha=0.35), plt.Line2D([], [], color=CORES["ngrama"], lw=1.0, ls=(0, (4, 3)))]
    rotulos += ["Pareto frontier, the floor entering at near-zero GPU cost", "6-gram floor (seconds on one CPU core)"]
    fig.legend(alcas, rotulos, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=1, fontsize=8, handlelength=2.2)
    salvar(fig, saida / "fig_frontier.pdf")


MODOS = ("autoregressivo", "difusao")
OBRAS = {"braz_cubas": "Brás Cubas", "dom_casmurro": "Dom Casmurro", "esau_e_jacob": "Esaú e Jacó", "helena": "Helena",
         "historias_sem_data": "Histórias sem data", "mao_e_luva": "A mão e a luva", "memorial_de_ayres": "Memorial de Aires",
         "quincas_borba": "Quincas Borba", "yaya_garcia": "Iaiá Garcia"}


def analises(corpus: str) -> dict:
    return ler("m7_analises.json" if corpus == "A" else f"m7_analises_{corpus}.json")


def run(corpus: str, modo: str, ep: int, s: int) -> Path:
    return RAIZ / "runs" / f"m7{'' if corpus == 'A' else '-' + corpus}-{modo}-e{ep}-s{s}"


def memorizacao(saida: Path):
    """Training against validation bits per byte across the budget grid, both corpora, mean over seeds."""
    fig, eixos = plt.subplots(1, 2, figsize=(8.0, 3.2), gridspec_kw={"wspace": 0.25})
    for ax, corpus in zip(eixos, ("A", "B")):
        tv = analises(corpus)["treino_validacao"]
        for modo in MODOS:
            for parte, estilo, mfc in (("validacao", "-", None), ("treino", "--", "white")):
                e, b = zip(*tv[modo][parte])
                ax.plot(e, b, ls=estilo, marker="o", ms=4, lw=1.5, color=CORES[modo], mfc=mfc or CORES[modo])
        a256 = dict(tv["autoregressivo"]["treino"])[256]
        ax.annotate(f"{a256:.3f}", (256, a256), xytext=(-6, 2), textcoords="offset points", ha="right", va="bottom",
                    fontsize=7.5, color=CORES["autoregressivo"])
        ax.set(xscale="log", xlabel="epochs over the same text (log scale)", title=f"corpus {corpus}")
        ax.set_xticks([1, 4, 16, 64, 256], ["1", "4", "16", "64", "256"])
        ax.minorticks_off()
        ax.title.set_fontsize(9)
        seta(ax, (0.20, 0.45), (0.20, 0.22))
    eixos[0].set_ylabel("bits per byte")
    alcas = [plt.Line2D([], [], color=CORES[m], lw=1.5) for m in MODOS] + \
            [plt.Line2D([], [], color=INK, lw=1.5, marker="o", ms=4), plt.Line2D([], [], color=INK, lw=1.5, ls="--", marker="o", ms=4, mfc="white")]
    fig.legend(alcas, [NOME[m] for m in MODOS] + ["validation", "training partition"], loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=8.5)
    salvar(fig, saida / "fig_memorisation.pdf")


def teste(saida: Path):
    """Delta = bpb_dif - bpb_AR on the test partition at 256 epochs, per work (A) and per group (B), with the
    interval over the 6 seeds per group and the two-level bootstrap for the total."""
    fig, eixos = plt.subplots(1, 2, figsize=(8.2, 3.9), gridspec_kw={"wspace": 0.55})
    for ax, corpus in zip(eixos, ("A", "B")):
        t = analises(corpus)["teste"]
        po = t["por_obra"]
        nomes = sorted(po, key=lambda g: po[g][0])
        rot = [OBRAS.get(g, g.replace("lei-", "Law ")) for g in nomes] + ["total"]
        vals = [po[g] for g in nomes] + [[t["delta"], *t["ic95_bootstrap"]]]
        for i, (d, lo, hi) in enumerate(vals):
            total = i == len(vals) - 1
            ax.plot([lo, hi], [i, i], color=INK, lw=1.6)
            ax.plot(d, i, "s" if total else "o", color=INK, ms=6 if total else 4.5)
        ax.axvline(0, color=MUTED, lw=1.0, ls=(0, (4, 3)))
        ax.set_yticks(range(len(rot)), rot, fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel(r"$\Delta$ bits per byte, test partition", fontsize=9)
        ax.set_title(f"corpus {corpus}, 256 epochs, 6 seeds", fontsize=9, loc="left")
        # zero at the centre, so both directions have room and the empty side is visible
        lim = 1.12 * max(abs(v) for d in vals for v in d)
        ax.set_xlim(-lim, lim)
        seta_l, rotulo_l = offset_copy_axes(ax, fig, -0.58), offset_copy_axes(ax, fig, -0.72)
        for sentido, texto in ((-1, "diffusion better"), (1, "autoregressive better")):
            ponta = 0.5 + sentido * 0.40
            ax.annotate("", xy=(ponta, 0), xytext=(0.5 + sentido * 0.02, 0), xycoords=seta_l, textcoords=seta_l,
                        annotation_clip=False, arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.7", color=INK, lw=1.4))
            ax.annotate(texto, xy=(0.5 + sentido * 0.21, 0), xycoords=rotulo_l, fontsize=8, color=INK, ha="center", va="top",
                        annotation_clip=False)
    salvar(fig, saida / "fig_test.pdf")


def offset_copy_axes(ax, fig, polegadas):
    from matplotlib.transforms import offset_copy
    return offset_copy(ax.transAxes, fig=fig, y=polegadas, units="inches")


def calibracao(saida: Path):
    """Left, reliability of the first decisive seed of corpus A, at temperature 1 and fitted. Right, ECE across
    the budget grid, mean over seeds, both corpora."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.0, 3.5), gridspec_kw={"wspace": 0.3, "width_ratios": [1, 1.25]})
    a1.plot([0, 1], [0, 1], color=MUTED, lw=1.0)
    for modo in MODOS:
        c = json.loads((run("A", modo, 256, 300) / "calibracao.json").read_text(encoding="utf-8"))
        for chave, estilo in (("T1", "--"), ("calibrado", "-")):
            r = c["confiabilidade"][chave]
            ok = [i for i, n in enumerate(r["n"]) if n >= 20]
            a1.plot(np.array(r["confianca"])[ok], np.array(r["acuracia"])[ok], ls=estilo, marker="o", ms=3.5, lw=1.4,
                    color=CORES[modo], mfc="white" if chave == "T1" else CORES[modo])
    a1.text(0.97, 0.05, "below the diagonal:\noverconfident", fontsize=7.5, color=MUTED, ha="right", va="bottom")
    a1.set(xlim=(0, 1), ylim=(0, 1), xlabel="declared confidence", ylabel="observed accuracy")
    a1.set_title("corpus A, 256 epochs, seed 300", fontsize=9, loc="left")
    seta(a1, (0.22, 0.78), (0.36, 0.64))
    for corpus, mk in (("A", "o"), ("B", "s")):
        cd = analises(corpus)["calibracao_delecao"]
        for modo in MODOS:
            eps = sorted(cd[modo], key=int)
            x = [int(e) for e in eps]
            a2.plot(x, [cd[modo][e]["ece_T1"][0] for e in eps], ls="--", marker=mk, ms=4, lw=1.3, color=CORES[modo],
                    mfc="white", alpha=1 if corpus == "A" else 0.6)
            a2.plot(x, [cd[modo][e]["ece_calibrado"][0] for e in eps], ls="-", marker=mk, ms=4, lw=1.3, color=CORES[modo],
                    alpha=1 if corpus == "A" else 0.6)
    a2.set(xscale="log", xlabel="epochs over the same text (log scale)", ylabel="expected calibration error")
    a2.set_xticks([1, 4, 16, 64, 256], ["1", "4", "16", "64", "256"])
    a2.minorticks_off()
    a2.set_title("mean over seeds, circles corpus A, squares corpus B", fontsize=9, loc="left")
    seta(a2, (0.12, 0.80), (0.12, 0.56))
    alcas = [plt.Line2D([], [], color=CORES[m], lw=1.4) for m in MODOS] + \
            [plt.Line2D([], [], color=INK, ls="--", lw=1.4, marker="o", ms=3.5, mfc="white"), plt.Line2D([], [], color=INK, lw=1.4, marker="o", ms=3.5)]
    fig.legend(alcas, [NOME[m] for m in MODOS] + ["temperature 1", "fitted temperature"], loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=8.5)
    salvar(fig, saida / "fig_calibration.pdf")


def delecao(saida: Path):
    """Left, deletion curves of the first decisive seed of corpus A, words removed by importance and at random.
    Right, probability of the decision with the context removed, relative to the full context, across budgets."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.0, 3.4), gridspec_kw={"wspace": 0.3})
    for modo in MODOS:
        d = json.loads((run("A", modo, 256, 300) / "delecao.json").read_text(encoding="utf-8"))
        a1.plot(d["grade"], d["curva_importancia_media"], marker="o", ms=3.5, lw=1.5, color=CORES[modo])
        a1.plot(d["grade"], d["curva_aleatoria_media"], ls="--", lw=1.3, color=CORES[modo])
    a1.set(xlabel="fraction of context words removed", ylabel="p(decision) relative to full context", ylim=(0, 1.05))
    a1.set_title("corpus A, 256 epochs, seed 300", fontsize=9, loc="left")
    seta(a1, (0.22, 0.80), (0.22, 0.60), "more faithful\nimportance")
    for corpus, mk in (("A", "o"), ("B", "s")):
        cd = analises(corpus)["calibracao_delecao"]
        for modo in MODOS:
            eps = sorted(cd[modo], key=int)
            a2.plot([int(e) for e in eps], [cd[modo][e]["p_vazio_relativo"][0] for e in eps], marker=mk, ms=4, lw=1.4,
                    color=CORES[modo], ls="-" if corpus == "A" else "--")
    a2.set(xscale="log", xlabel="epochs over the same text (log scale)", ylabel="p(decision) with empty context, relative")
    a2.set_xticks([1, 4, 16, 64, 256], ["1", "4", "16", "64", "256"])
    a2.minorticks_off()
    a2.set_title("mean over seeds", fontsize=9, loc="left")
    seta(a2, (0.88, 0.72), (0.88, 0.48), "uses more\ncontext")
    alcas = [plt.Line2D([], [], color=CORES[m], lw=1.5) for m in MODOS] + \
            [plt.Line2D([], [], color=INK, lw=1.5, marker="o", ms=3.5), plt.Line2D([], [], color=INK, lw=1.3, ls="--")]
    fig.legend(alcas, [NOME[m] for m in MODOS] + ["by importance (left), corpus A (right)", "at random (left), corpus B (right)"],
               loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=8.5)
    salvar(fig, saida / "fig_deletion.pdf")


def sensibilidade(saida: Path):
    """Validation bits per byte at 256 epochs on corpus A, one factor at a time, seeds 300 to 302 as points
    and their mean as a line; the default value of each factor is marked."""
    s = ler("sensibilidade.json")["fatores"]
    paineis = (("L", "window $L$ (tokens)", True), ("V", "vocabulary $V$", True), ("precisao", "training precision", False),
               ("K", "NELBO draws $K$", True))
    fig, eixos = plt.subplots(1, 4, figsize=(9.6, 2.9), sharey=True, gridspec_kw={"wspace": 0.12})
    padrao = {"L": "256", "V": "4096", "precisao": "fp16", "K": "1"}
    for ax, (f, nome, log) in zip(eixos, paineis):
        linhas = s[f]
        chaves = sorted(linhas, key=lambda v: float(v) if v[0].isdigit() else {"fp16": 16, "fp32": 32}[v])
        x = [float(v) if log else i for i, v in enumerate(chaves)] if log else list(range(len(chaves)))
        for modo in MODOS:
            chave = f"bpb_{modo}"
            if chave not in linhas[chaves[0]]:
                continue
            for xi, v in zip(x, chaves):
                ax.scatter([xi] * len(linhas[v][chave]["por_semente"]), linhas[v][chave]["por_semente"], s=12,
                           color=CORES[modo], alpha=0.5, lw=0)
            ax.plot(x, [linhas[v][chave]["media"] for v in chaves], marker="o", ms=4, lw=1.5, color=CORES[modo])
        if f == "K":
            ar = s["L"]["256"]["bpb_autoregressivo"]["media"]
            ax.axhline(ar, color=CORES["autoregressivo"], lw=1.2, ls=(0, (4, 3)))
            ax.text(0.97, ar - 0.04, "autoregressive,\nexact (no $K$)", transform=ax.get_yaxis_transform(), ha="right", va="top",
                    fontsize=7.5, color=CORES["autoregressivo"])
        ax.axvline(x[chaves.index(padrao[f])], color=MUTED, lw=0.9, ls=":")
        if log:
            ax.set_xscale("log", base=2)
        ax.set_xticks(x, chaves, fontsize=8)
        ax.minorticks_off()
        ax.set_xlabel(nome, fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        seta(ax, (0.12, 0.62), (0.12, 0.40))
    eixos[0].set_ylabel("validation bits per byte")
    alcas = [plt.Line2D([], [], color=CORES[m], lw=1.5, marker="o", ms=4) for m in MODOS] + [plt.Line2D([], [], color=MUTED, lw=0.9, ls=":")]
    fig.legend(alcas, [NOME[m] for m in MODOS] + ["value used in the comparison"], loc="upper center",
               bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8.5)
    salvar(fig, saida / "fig_sensitivity.pdf")


def entropia(saida: Path):
    """Mean entropy of the samples against their generative perplexity under Tucano 630M, per sampling
    temperature, at 16 and 256 epochs on corpus A."""
    eg = ler("entropia_genppl.json")
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for chave, pts in eg.items():
        modo, ep = chave.split("_e")
        h, g, t = [p["entropia"] for p in pts], [p["genppl"] for p in pts], [p["temperatura"] for p in pts]
        ax.plot(h, g, marker="o", ms=4, lw=1.5, color=CORES[modo], ls="-" if ep == "256" else "--",
                mfc=CORES[modo] if ep == "256" else "white", label=f"{NOME[modo]}, {ep} epochs")
        for hi, gi, ti in zip(h, g, t):
            ax.annotate(f"{ti:g}", (hi, gi), xytext=(4, -3), textcoords="offset points", fontsize=7, color=MUTED, va="top")
    ax.set(yscale="log", xlabel="mean entropy of a sample (nats, over its own tokens)",
           ylabel="generative perplexity, Tucano 630M")
    ax.minorticks_off()
    seta(ax, (1.04, 0.62), (1.04, 0.38), "more probable\ntext")
    seta(ax, (0.55, -0.26), (0.85, -0.26), "more diverse")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.33), ncol=2, fontsize=8)
    salvar(fig, saida / "fig_entropy.pdf")


def tabela_calibracao(saida: Path):
    """Rows of the per-budget calibration and deletion table, mean over the seeds of each budget."""
    nomes = {"autoregressivo": "AR", "difusao": "Diff."}
    linhas = []
    for corpus in ("A", "B"):
        cd = analises(corpus)["calibracao_delecao"]
        for modo in MODOS:
            for e in sorted(cd[modo], key=int):
                r = cd[modo][e]
                pb = r["portao_beta"]["0.9"]
                linhas.append(f"{corpus} & {nomes[modo]} & {e} & {r['T'][0]:.2f} & {r['ece_T1'][0]:.3f} & {r['ece_calibrado'][0]:.3f} & "
                              f"{r['auc_roc'][0]:.3f} & {pb['cobertura'][0]:.3f} & {pb['promessa_cumprida']}/{pb['n']} & "
                              f"{r['auc_delecao_importancia'][0]:.3f} & {r['auc_delecao_aleatoria'][0]:.3f} \\\\")
            if modo == "autoregressivo":
                linhas.append("\\addlinespace")
        if corpus == "A":
            linhas.append("\\midrule")
    (saida / "calibration.tex").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def tabela_tarefas(saida: Path):
    """Rows of the task table: estimate and 95% interval per scorer on the external set, for all acts
    and by genre (ordinary laws and decrees), each block headed by its number of groups per task."""
    g = ler("m14_B.json")["gerativa"]
    nomes = {"ngrama": "6-gram floor", **{m: n for m, (n, _) in PONTUADORES.items()}}
    cel = lambda v: f"{v['estimativa']:.3f} [{v['ic95'][0]:.3f}, {v['ic95'][1]:.3f}]"
    blocos = {"All acts": lambda t, m: g[t][m], "Ordinary laws": lambda t, m: g[t][m]["por_genero"]["lei"],
              "Decrees": lambda t, m: g[t][m]["por_genero"]["dec"]}
    linhas = []
    for k, (rot, f) in enumerate(blocos.items()):
        if k:
            linhas.append(r"\midrule")
        linhas.append(rf"\emph{{{rot}}} & " + " & ".join(rf"\emph{{{f(t, 'ngrama')['grupos']} groups}}" for t in TAREFAS) + r" \\")
        linhas += [f"{nome} & " + " & ".join(cel(f(t, m)) for t in TAREFAS) + r" \\" for m, nome in nomes.items()]
    (saida / "tasks.tex").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def tabela_diagnostico(saida: Path):
    """Rows of the surface-frequency diagnostic of T1 and T3 (scripts/diagnostico_frequencia.py):
    item accuracy by class and genre, rate of choosing the candidate most frequent in the training
    text of corpus B, and accuracy in the decrees that form a group of one item."""
    d = ler("diagnostico_frequencia_B.json")
    nomes = {"ngrama": "6-gram floor", **{m: n for m, (n, _) in PONTUADORES.items()}}
    linhas = []
    for m, nome in nomes.items():
        t1, t3 = d["t1"][m], d["t3"][m]
        v = [t1["por_classe"]["lei:numero"]["acerto"], t1["por_classe"]["lei:remissao"]["acerto"],
             t1["por_classe"]["dec:remissao"]["acerto"], t1["todos"]["escolhe_mais_frequente"],
             t3["todos"]["prefere_mais_frequente"], t1["dec_grupo_de_1"]["acerto"], t3["dec_grupo_de_1"]["acerto"],
             t3["dec_grupo_maior"]["acerto"]]
        linhas.append(f"{nome} & " + " & ".join(f"{x:.3f}" for x in v) + r" \\")
    (saida / "diagnostic.tex").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def tabela_decisor(saida: Path):
    """Rows of the typed decider table: balanced accuracy in T2, accuracy in T3, on the external set,
    and the selective answer at the 95% target, as mean coverage over seeds and the accuracy of the
    covered items pooled over seeds."""
    j = ler("m14_B.json")
    d, res, laya = j["decisor"], j["decisor_resumo"], j["laya_resumo"]
    nomes = {"tfidf": "TF-IDF with logistic regression (floor)", "autoregressivo": "autoregressive checkpoint",
             "difusao": "masked diffusion checkpoint", "laya": "Laya multilingual, fine-tuned",
             "laya-sem-ajuste": "Laya multilingual, no fine-tuning"}

    def seletiva(m, t):
        v = [r for k, r in res.items() if k.startswith(f"m7-B-{m}") and k.endswith(t)] if m in ("autoregressivo", "difusao") \
            else [laya[t]] if m == "laya" else []
        if not v:
            return ""
        cob = np.array([r["cobertura_externo"] for r in v])
        ac = sum(r["cobertura_externo"] * (r["acerto_seletivo_externo"] or 0) for r in v)
        return f"{100 * cob.mean():.2f}\\%" + (f" ({ac / cob.sum():.2f})" if cob.sum() else "")

    linhas = []
    for m, nome in nomes.items():
        cel = [f"{d[t][m]['estimativa']:.3f} [{d[t][m]['ic95'][0]:.3f}, {d[t][m]['ic95'][1]:.3f}]" for t in ("t2", "t3")]
        linhas.append(f"{nome} & " + " & ".join(cel) + " & " + " & ".join(seletiva(m, t) for t in ("t2", "t3")) + r" \\")
    (saida / "decider.tex").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def tabela_busca(saida: Path):
    """Rows of the hyperparameter search table: mean validation bpb of the 2 seeds per configuration."""
    def media(modo, lr, do, ep):
        return np.mean([json.loads((RAIZ / "runs" / f"hp-{modo}-lr{lr:g}-do{do:g}-e{ep}-s{s}" / "avaliacao.json")
                                   .read_text(encoding="utf-8"))["bpb"] for s in (0, 1)])
    esc = json.loads((RAIZ / "preregistro" / "m6_delta_min.json").read_text(encoding="utf-8"))["escolha"]
    linhas = []
    for lr in (3e-4, 1e-3, 3e-3):
        for do in (0.0, 0.1, 0.2):
            cel = []
            for modo in ("autoregressivo", "difusao"):
                for ep in (16, 256):
                    v = f"{media(modo, lr, do, ep):.3f}"
                    cel.append(rf"\textbf{{{v}}}" if ep == 256 and esc[modo] == {"dropout": do, "lr": lr} else v)
            linhas.append(rf"$3 \times 10^{{{int(np.log10(lr / 3))}}}$" if lr != 1e-3 else r"$10^{-3}$")
            linhas[-1] += f" & {do:g} & " + " & ".join(cel) + r" \\"
    (saida / "hp_search.tex").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def tabela_bytes(saida: Path):
    """Rows of the byte-level control at equal compute: BPE reference (B1) and the trivial byte
    tokenizer (B0) with the window in symbols or in bytes, autoregressive mode, 3 seeds, difference
    paired by seed with a t interval on 2 degrees of freedom."""
    from scipy import stats

    def corrida(braco, janela, ep, s):
        r = RAIZ / "runs" / f"m10-{braco}-{janela}-e{ep}-s{s}-aut"
        return (json.loads((r / "config.json").read_text(encoding="utf-8")),
                json.loads((r / "avaliacao.json").read_text(encoding="utf-8"))["bpb"])

    linhas = []
    for ep in (16, 256):
        ref = np.array([corrida("B1", "simbolos", ep, s)[1] for s in range(3)])
        linhas.append(rf"BPE, $V = 4096$ & {ep} & 256 & {ref.mean():.3f} & \\")
        for janela in ("simbolos", "bytes"):
            cfg, _ = corrida("B0", janela, ep, 0)
            v = np.array([corrida("B0", janela, ep, s)[1] for s in range(3)])
            d = v - ref
            h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            linhas.append(rf"Bytes, $V = 260$ & {cfg['epocas']:.1f} & {cfg['L']} & {v.mean():.3f} & "
                          rf"${d.mean():+.3f}$ $[{d.mean() - h:+.3f}, {d.mean() + h:+.3f}]$ \\")
        if ep == 16:
            linhas.append(r"\midrule")
    (saida / "bytes_control.tex").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", type=Path, required=True)
    ap.add_argument("--tabelas", type=Path)
    a = ap.parse_args()
    a.saida.mkdir(parents=True, exist_ok=True)
    for figura in (epocas, sentinela, onde, mistura, epocas_b, curva, tarefas, fronteira, memorizacao, teste, calibracao, delecao,
                   sensibilidade, entropia):
        figura(a.saida)
    if a.tabelas:
        a.tabelas.mkdir(parents=True, exist_ok=True)
        for tabela in (tabela_busca, tabela_bytes, tabela_tarefas, tabela_diagnostico, tabela_decisor, tabela_calibracao):
            tabela(a.tabelas)


if __name__ == "__main__":
    main()

"""Figures for training, validation, test, result analysis, sensitivity analysis and deletion curves.

Every function takes plain data (lists, dicts, arrays read from runs/ and analysis files) and
returns a matplotlib Figure. Labels are in Portuguese, the two modes keep fixed colours across
figures, and `salvar` writes PNG for documents. Every figure carries an arrow toward the better
direction of its metric, or, where the metric has no better side, toward what each side means. The backend is left to the caller: scripts
select Agg, a notebook keeps its inline backend.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import offset_copy

CORES = {"autoregressivo": "#1f5a7a", "difusao": "#a15a1f"}
ROTULO = {"autoregressivo": "autorregressivo", "difusao": "difusão"}
# labels of the figures the notebooks draw, in Portuguese (default) and English
TEXTOS = {
    "pt": {"modos": ROTULO, "melhor": "melhor", "sub": "subconfiança", "super": "superconfiança",
           "conf_x": "confiança declarada", "conf_y": "acurácia observada",
           "conf_titulo": "Confiança declarada contra acurácia observada", "importancia": "importância",
           "aleatoria": "aleatória", "rho": "ρ de Spearman", "del_x": "fração de palavras removidas",
           "del_y": "p(decisão) / p0", "del_titulo": "Curva de deleção (em 100%: viés intrínseco do modelo)",
           "p0_x": "probabilidade calibrada p0", "p0_y": "AUC da curva de deleção",
           "p0_titulo": "Confiança contra dependência do contexto", "fiel": "importância mais fiel",
           "dependente": "mais dependente do contexto"},
    "en": {"modos": {"autoregressivo": "autoregressive", "difusao": "diffusion"}, "melhor": "better",
           "sub": "underconfidence", "super": "overconfidence", "conf_x": "stated confidence",
           "conf_y": "observed accuracy", "conf_titulo": "Stated confidence against observed accuracy",
           "importancia": "importance", "aleatoria": "random", "rho": "Spearman's ρ",
           "del_x": "share of words removed", "del_y": "p(decision) / p0",
           "del_titulo": "Deletion curve (at 100%: the model's intrinsic bias)", "p0_x": "calibrated probability p0",
           "p0_y": "AUC of the deletion curve", "p0_titulo": "Confidence against dependence on context",
           "fiel": "more faithful importance", "dependente": "more context-dependent"},
}


TINTA = "#14213D"


def seta(ax, inicio, fim, texto="melhor"):
    """Arrow from worse to better in axes fraction, as in the article figures; the label sits beside a
    vertical arrow and above a horizontal one, so it never lies on the shaft."""
    ax.annotate("", xy=fim, xytext=inicio, xycoords="axes fraction", annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.7", color=TINTA, lw=1.4))
    vertical = abs(fim[1] - inicio[1]) >= abs(fim[0] - inicio[0])
    meio = ((inicio[0] + fim[0]) / 2, (inicio[1] + fim[1]) / 2)
    ax.annotate(texto, xy=meio, xytext=(6, 0) if vertical else (0, 6), textcoords="offset points",
                xycoords="axes fraction", fontsize=9, color=TINTA, annotation_clip=False,
                ha="left" if vertical else "center", va="center" if vertical else "bottom")


def _cor(modo: str) -> str:
    return CORES.get(modo, "#555555")


def salvar(fig, caminho: Path, dpi: int = 200) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return caminho


def _ic(amostras, nivel=0.95):
    a = np.asarray(amostras, dtype=float)
    q = (1 - nivel) / 2
    return a.mean(), np.quantile(a, q), np.quantile(a, 1 - q)


# Training procedure

def treino(passos: dict) -> plt.Figure:
    """passos: {modo: [rows of passos.jsonl]}. Loss, learning rate, GradScaler scale and skipped steps."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for modo, linhas in passos.items():
        x = [r["passo"] for r in linhas]
        ax[0].plot(x, [r["perda"] for r in linhas], color=_cor(modo), label=ROTULO.get(modo, modo))
        ax[1].plot(x, [r["taxa"] for r in linhas], color=_cor(modo))
        ax[2].plot(x, [r["escala_gradscaler"] for r in linhas], color=_cor(modo))
        ax2 = ax[2].twinx()
        ax2.step(x, [r["passos_pulados"] for r in linhas], color=_cor(modo), ls=":", where="post")
        ax2.set_ylabel("passos pulados (acumulado, pontilhado)")
    ax[0].set(title="Perda de treino", xlabel="passo aplicado", ylabel="perda (nats por token)")
    ax[1].set(title="Taxa de aprendizado", xlabel="passo aplicado")
    ax[2].set(title="Escala do GradScaler", xlabel="passo aplicado", yscale="log")
    seta(ax[0], (0.80, 0.80), (0.80, 0.55))
    seta(ax[1], (0.45, 0.92), (0.75, 0.92), "aquecimento, depois cosseno")
    seta(ax[2], (0.06, 0.30), (0.06, 0.08), "queda: estouro em fp16")
    ax[0].legend()
    fig.tight_layout()
    return fig


def sentinela(linhas: list[dict], limite: float = 0.05) -> plt.Figure:
    """Worst per-tensor relative gradient error (p95) per band of l/L along training, and underflow."""
    faixas = sorted({r["faixa_l"] for r in linhas})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for f in faixas:
        sel = [r for r in linhas if r["faixa_l"] == f]
        passos = sorted({r["passo"] for r in sel})
        pior = [max((r["erro_relativo_p95"] or 0) for r in sel if r["passo"] == p) for p in passos]
        under = [max(r["underflow"] for r in sel if r["passo"] == p) for p in passos]
        ax[0].plot(passos, pior, marker="o", label=f)
        ax[1].plot(passos, under, marker="o", label=f)
    ax[0].axhline(limite, color="k", ls="--", lw=1, label="margem")
    ax[1].axhline(0.01, color="k", ls="--", lw=1)
    ax[0].set(title="Erro relativo do gradiente fp16 (pior tensor, p95)", xlabel="passo", yscale="log")
    ax[1].set(title="Underflow (pior tensor)", xlabel="passo")
    seta(ax[0], (0.45, 0.85), (0.45, 0.62))
    seta(ax[1], (0.45, 0.85), (0.45, 0.62))
    ax[0].legend(title="faixa de l/L", fontsize=8)
    fig.tight_layout()
    return fig


# Validation

def validacao_epocas(dados: dict, confirmatorio: float | None = 256) -> plt.Figure:
    """dados: {modo: {epocas: [bpb per seed]}}. Mean and 95% interval across seeds, log epochs."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for modo, por_ep in dados.items():
        eps = sorted(por_ep)
        m, lo, hi = zip(*(_ic(por_ep[e]) for e in eps))
        ax.plot(eps, m, marker="o", color=_cor(modo), label=ROTULO.get(modo, modo))
        ax.fill_between(eps, lo, hi, color=_cor(modo), alpha=0.2)
    if confirmatorio:
        ax.axvline(confirmatorio, color="k", ls=":", lw=1)
    ax.set(xscale="log", xlabel="épocas sobre o mesmo texto", ylabel="bits por byte (validação)",
           title="Validação ao longo do orçamento")
    seta(ax, (0.06, 0.45), (0.06, 0.20))
    ax.legend()
    return fig


def treino_validacao(curvas: dict) -> plt.Figure:
    """curvas: {modo: {"treino": [(epoca, bpb)], "validacao": [(epoca, bpb)]}}. The gap signals memorisation."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for modo, c in curvas.items():
        for parte, estilo in (("treino", "--"), ("validacao", "-")):
            e, b = zip(*c[parte])
            ax.plot(e, b, ls=estilo, color=_cor(modo), label=f"{ROTULO.get(modo, modo)}, {parte}")
    ax.set(xscale="log", xlabel="épocas", ylabel="bits por byte", title="Treino contra validação")
    seta(ax, (0.06, 0.45), (0.06, 0.20))
    ax.legend(fontsize=8)
    return fig


# Test and results

def delta_por_obra(por_obra: dict, ic_total: tuple, rotulo_total: str = "total") -> plt.Figure:
    """Forest plot of Delta = bpb_dif - bpb_AR. por_obra: {obra: (delta, lo, hi)}; ic_total: (delta, lo, hi)."""
    nomes = list(por_obra) + [rotulo_total]
    vals = list(por_obra.values()) + [ic_total]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(nomes) + 1))
    for i, (d, lo, hi) in enumerate(vals):
        ax.plot([lo, hi], [i, i], color="k")
        ax.plot(d, i, "s" if i == len(vals) - 1 else "o", color="k", ms=8 if i == len(vals) - 1 else 5)
    ax.axvline(0, color="#888", ls="--")
    ax.set_yticks(range(len(nomes)), nomes)
    ax.invert_yaxis()
    ax.set(xlabel="Δ bits por byte (difusão menos autoregressivo)",
           title="Abaixo de zero: difusão melhor. Acima: o AR supera o limite da difusão")
    x0, x1 = ax.get_xlim()
    z = (0 - x0) / (x1 - x0)
    # two arrows leaving zero, a fixed distance below the axis label, each labelled at its head
    linha = offset_copy(ax.transAxes, fig=fig, y=-0.75, units="inches")
    for sentido, texto in ((-1, "difusão melhor"), (1, "autorregressivo melhor")):
        ponta = (z + sentido * 0.22, 0)
        ax.annotate("", xy=ponta, xytext=(z + sentido * 0.01, 0), xycoords=linha, textcoords=linha, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.7", color=TINTA, lw=1.4))
        ax.annotate(texto, xy=ponta, xycoords=linha, xytext=(sentido * 4, 0), textcoords="offset points", fontsize=9,
                    color=TINTA, ha="left" if sentido > 0 else "right", va="center", annotation_clip=False)
    return fig


def com_sem_nomes(dados: dict) -> plt.Figure:
    """dados: {modo: (bpb, bpb_sem_nomes)}."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(dados))
    for j, (rot, idx) in enumerate((("com nomes", 0), ("sem nomes", 1))):
        ax.bar(x + (j - 0.5) * 0.35, [v[idx] for v in dados.values()], 0.35, label=rot)
    ax.set_xticks(x, [ROTULO.get(m, m) for m in dados])
    ax.set(ylabel="bits por byte (teste)", title="Efeito dos nomes próprios vistos no treino")
    seta(ax, (1.03, 0.60), (1.03, 0.35))
    ax.legend()
    return fig


def novidade_ngramas(dados: dict) -> plt.Figure:
    """dados: {rotulo: [fraction of novel n-grams for n = 1..10]}, including the human test text."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for rot, fr in dados.items():
        ax.plot(range(1, len(fr) + 1), fr, marker="o", label=rot, color=_cor(rot) if rot in CORES else None,
                ls="--" if "humano" in rot else "-")
    ax.set(xlabel="n", ylabel="fração de n-gramas ausentes do treino", ylim=(0, 1),
           title="Novidade: gera ou copia? Melhor: perto do texto humano")
    seta(ax, (1.03, 0.55), (1.03, 0.85), "mais invenção")
    seta(ax, (1.03, 0.45), (1.03, 0.15), "mais cópia do treino")
    ax.legend()
    return fig


def descarte_copia(dados: dict) -> plt.Figure:
    """dados: {modo: {epocas: (taxa, lo, hi)}}. Share of samples discarded by the copy filter."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for modo, por_ep in dados.items():
        eps = sorted(por_ep)
        t, lo, hi = zip(*(por_ep[e] for e in eps))
        ax.errorbar(eps, t, yerr=[np.subtract(t, lo), np.subtract(hi, t)], marker="o", capsize=3,
                    color=_cor(modo), label=ROTULO.get(modo, modo))
    ax.set(xscale="log", xlabel="épocas", ylabel="fração descartada por cópia", title="Memorização ao longo do treino")
    seta(ax, (1.03, 0.60), (1.03, 0.35))
    ax.legend()
    return fig


def entropia_genppl(dados: dict) -> plt.Figure:
    """dados: {rotulo: [(temperatura, entropia, perplexidade gerativa)]}."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for rot, pts in dados.items():
        t, h, g = zip(*pts)
        ax.plot(h, g, marker="o", label=rot, color=_cor(rot) if rot in CORES else None)
        for ti, hi, gi in pts:
            ax.annotate(f"{ti:g}", (hi, gi), fontsize=7, textcoords="offset points", xytext=(3, 3))
    ax.set(xlabel="entropia das amostras", ylabel="perplexidade gerativa", yscale="log",
           title="Qualidade contra diversidade (rótulo: temperatura)")
    seta(ax, (1.03, 0.60), (1.03, 0.35), "texto mais provável")
    seta(ax, (0.55, -0.22), (0.85, -0.22), "mais diversidade")
    ax.legend()
    return fig


def niveis(dados: dict) -> plt.Figure:
    """dados: {nivel: {modo: {epocas: (proporcao, limite_inferior)}}}."""
    fig, axs = plt.subplots(1, len(dados), figsize=(4.5 * len(dados), 4), sharey=True, squeeze=False)
    for ax, (nv, por_modo) in zip(axs[0], dados.items()):
        for modo, por_ep in por_modo.items():
            eps = sorted(por_ep)
            p, lo = zip(*(por_ep[e] for e in eps))
            ax.plot(eps, p, marker="o", color=_cor(modo), label=ROTULO.get(modo, modo))
            ax.plot(eps, lo, ls=":", color=_cor(modo))
        ax.axhline(10 / 12, color="#888", ls="--", lw=1)
        ax.set(xscale="log", title=nv, xlabel="épocas", ylim=(0, 1.02))
        seta(ax, (0.88, 0.25), (0.88, 0.50))
    axs[0][0].set_ylabel("itens com mediana ≥ 2 (pontilhado: limite inferior)")
    axs[0][0].legend()
    fig.tight_layout()
    return fig


# Calibration and selective decision (criteria of the typed-decision class)

def confiabilidade(sistemas: dict, n_minimo: int = 20, idioma: str = "pt") -> plt.Figure:
    """sistemas: {rotulo: dict from calibracao.confiabilidade, with optional 'ece'}. Above the
    diagonal is under-confidence, below it over-confidence. Bins with fewer than n_minimo
    predictions are omitted and marker area grows with the bin count. Labels containing
    'T=1' (uncalibrated) are dashed. idioma picks the language of the labels."""
    tx = TEXTOS[idioma]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.fill_between([0, 1], [0, 1], [1, 1], color="#e6f4d8", alpha=0.5, label=tx["sub"])
    ax.fill_between([0, 1], [0, 0], [0, 1], color="#f4e6d8", alpha=0.5, label=tx["super"])
    ax.plot([0, 1], [0, 1], color="k", lw=1)
    for rot, c in sistemas.items():
        ok = [i for i, n in enumerate(c["n"]) if n >= n_minimo]
        x, y, n = (np.asarray(c[k], dtype=float)[ok] for k in ("confianca", "acuracia", "n"))
        modo = rot.split()[0]
        cor = _cor(modo) if modo in CORES else None
        nome = rot.replace(modo, tx["modos"].get(modo, modo), 1)
        rotulo = f"{nome} (ECE {c['ece']:.3f})" if "ece" in c else nome
        ax.plot(x, y, ls="--" if "T=1" in rot else "-", color=cor, label=rotulo)
        ax.scatter(x, y, s=10 + 90 * np.sqrt(n / max(n.max(), 1)), color=cor, zorder=3)
    ax.set(xlabel=tx["conf_x"], ylabel=tx["conf_y"], xlim=(0, 1), ylim=(0, 1), title=tx["conf_titulo"])
    seta(ax, (0.13, 0.60), (0.29, 0.44), tx["melhor"])
    ax.legend(fontsize=8, loc="upper left")
    return fig


def seletiva(curvas: dict, alvo: float | None = None) -> plt.Figure:
    """curvas: {rotulo: dict from calibracao.curva_seletiva}. Coverage against selective accuracy and
    type I and II errors against coverage."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for rot, c in curvas.items():
        cor = _cor(rot.split()[0]) if rot.split()[0] in CORES else None
        ax[0].plot(c["cobertura"], c["acuracia"], label=rot, color=cor)
        ax[1].plot(c["cobertura"], c["erro_tipo_I"], label=f"{rot}, tipo I", color=cor)
        ax[1].plot(c["cobertura"], c["erro_tipo_II"], ls="--", label=f"{rot}, tipo II", color=cor)
    if alvo:
        ax[0].axhline(alvo, color="k", ls=":", lw=1)
    ax[0].set(xlabel="cobertura (fração aceita)", ylabel="acurácia entre os aceitos", title="Decisão seletiva")
    ax[1].set(xlabel="cobertura", ylabel="erros", title="Tipo I: aceito e errado. Tipo II: rejeitado e certo")
    seta(ax[0], (0.80, 0.45), (0.80, 0.70))
    seta(ax[1], (1.03, 0.60), (1.03, 0.35))
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    return fig


# Sensitivity analysis

def sensibilidade_grade(grade: dict, linhas: list, colunas: list, rot_l: str, rot_c: str) -> plt.Figure:
    """grade: {modo: 2-D array (len(linhas) x len(colunas)) of validation bpb}. One heat map per mode."""
    fig, axs = plt.subplots(1, len(grade), figsize=(5.5 * len(grade), 4.5), squeeze=False)
    for ax, (modo, M) in zip(axs[0], grade.items()):
        M = np.asarray(M, dtype=float)
        im = ax.imshow(M, cmap="viridis_r", aspect="auto")
        for (i, j), v in np.ndenumerate(M):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color="w")
        ax.set_xticks(range(len(colunas)), colunas)
        ax.set_yticks(range(len(linhas)), linhas)
        ax.set(xlabel=rot_c, ylabel=rot_l, title=ROTULO.get(modo, modo))
        cb = fig.colorbar(im, ax=ax, label="bits por byte (validação)")
        seta(cb.ax, (5.2, 0.75), (5.2, 0.25))
    fig.tight_layout()
    return fig


def sensibilidade_linha(dados: dict, parametro: str, metrica: str, xlog: bool = False, melhor: str = "baixo",
                        referencia: tuple | None = None) -> plt.Figure:
    """dados: {rotulo: {valor: [metric per seed or replicate]}}. Mean and 95% interval per value; `melhor`
    is "baixo" or "alto", and `referencia` an optional (value, label) drawn as a dashed line."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for rot, por_v in dados.items():
        vs = sorted(por_v)
        m, lo, hi = zip(*(_ic(por_v[v]) for v in vs))
        ax.plot(vs, m, marker="o", label=rot, color=_cor(rot) if rot in CORES else None)
        ax.fill_between(vs, lo, hi, alpha=0.2, color=_cor(rot) if rot in CORES else None)
    if referencia:
        ax.axhline(referencia[0], color="#888", ls="--", lw=1)
        ax.annotate(referencia[1], (0.02, referencia[0]), xycoords=("axes fraction", "data"), fontsize=8,
                    color="#555", va="bottom")
    ax.set(xlabel=parametro, ylabel=metrica, title=f"Sensibilidade a {parametro}", xscale="log" if xlog else "linear")
    if melhor == "baixo":
        seta(ax, (1.03, 0.60), (1.03, 0.35))
    else:
        seta(ax, (1.03, 0.35), (1.03, 0.60))
    ax.legend()
    return fig


# Deletion curves

def delecao(resultados: dict, idioma: str = "pt") -> plt.Figure:
    """resultados: {modo: output of delecao.analisar}. Mean curves by importance and at random, and
    the per-item scatter of calibrated p0 against the deletion AUC with Spearman's rho, labels in idioma."""
    tx = TEXTOS[idioma]
    nome = lambda modo: tx["modos"].get(modo, modo)  # noqa: E731
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for modo, r in resultados.items():
        g, cor = r["grade"], _cor(modo)
        ax[0].plot(g, r["curva_importancia_media"], marker="o", color=cor,
                   label=f"{nome(modo)}, {tx['importancia']} (AUC {r['auc_importancia_media']:.3f})")
        ax[0].plot(g, r["curva_aleatoria_media"], ls="--", color=cor,
                   label=f"{nome(modo)}, {tx['aleatoria']} (AUC {r['auc_aleatoria_media']:.3f})")
        p0 = [i["p0"] for i in r["itens"]]
        auc = [i["auc_importancia"] for i in r["itens"]]
        ax[1].scatter(p0, auc, s=10, alpha=0.5, color=cor,
                      label=f"{nome(modo)}, {tx['rho']} {r['spearman_p0_auc']['rho']:.2f}")
    ax[0].set(xlabel=tx["del_x"], ylabel=tx["del_y"], ylim=(0, 1.05), title=tx["del_titulo"])
    ax[1].set(xlabel=tx["p0_x"], ylabel=tx["p0_y"], title=tx["p0_titulo"])
    seta(ax[0], (1.03, 0.60), (1.03, 0.35), tx["fiel"])
    seta(ax[1], (1.03, 0.60), (1.03, 0.35), tx["dependente"])
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.75)
    return fig

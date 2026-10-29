#!/usr/bin/env python3
"""Build every figure of SPEC section 14, per corpus, and the summary of the per-run analyses.

Reads the M7 runs of the corpus (descriptive grid and confirmatory point), the M6 search, the
sensitivity results, the M9 samples and the entropy diagnostic. A figure whose inputs do not exist
yet is skipped and listed, so the script can be rerun as milestones complete. Writes the PNG files
to figuras/resultados/<corpus>/ and the numbers behind them to resultados/m7_analises[_B].json.

    python scripts/gerar_figuras.py --corpus A
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slm import estatistica as est  # noqa: E402
from slm import graficos  # noqa: E402
from slm import leitura as lt  # noqa: E402
from slm.corpus import RAIZ, ler_documentos  # noqa: E402

MODOS = ("autoregressivo", "difusao")
CONFIRMATORIO = 256
NOME_M7 = re.compile(r"^m7(-B)?-(autoregressivo|difusao)-e(\d+)-s(\d+)$")
NOME_HP = re.compile(r"^hp(-B)?-(autoregressivo|difusao)-lr([\d.e-]+)-do([\d.]+)-e(\d+)-s(\d+)$")


def ler_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def ler_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def intervalo_t(x) -> tuple:
    from scipy import stats
    x = np.asarray(x, dtype=float)
    m = float(x.mean())
    if x.size < 2:
        return m, m, m
    h = float(stats.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size))
    return m, m - h, m + h


def corridas_m7(corpus: str) -> dict:
    """{(modo, epocas): {semente: run folder}} of the M7 runs of the corpus."""
    saida = collections.defaultdict(dict)
    for p in sorted((RAIZ / "runs").iterdir()):
        m = NOME_M7.match(p.name)
        if m and (m.group(1) is not None) == (corpus == "B"):
            saida[(m.group(2), int(m.group(3)))][int(m.group(4))] = p
    return saida


def novidade(textos: list[str], treino: list[str], n_max: int = 10) -> list[float]:
    """Share of word n-grams of the texts absent from the training partition, n = 1..n_max."""
    pal_treino = [lt.palavras(t) for t in treino]
    pal = [lt.palavras(t) for t in textos]
    fr = []
    for n in range(1, n_max + 1):
        vistos = {hash(tuple(p[i:i + n])) for p in pal_treino for i in range(len(p) - n + 1)}
        gs = [hash(tuple(p[i:i + n])) for p in pal for i in range(len(p) - n + 1)]
        fr.append(float(np.mean([g not in vistos for g in gs])) if gs else float("nan"))
    return fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="A")
    a = ap.parse_args()
    corpus = a.corpus
    sufixo = "" if corpus == "A" else f"_{corpus}"
    saida = RAIZ / "figuras" / "resultados" / corpus
    m7 = corridas_m7(corpus)
    resumo, feitas, puladas = {}, [], []

    def fazer(nome, fig_fn):
        try:
            fig = fig_fn()
        except (KeyError, ValueError, IndexError, StopIteration, TypeError, FileNotFoundError) as e:
            puladas.append(f"{nome} ({type(e).__name__}: {e})")
            return
        if fig is None:
            puladas.append(f"{nome} (sem dados)")
            return
        feitas.append(str(graficos.salvar(fig, saida / f"{nome}.png").relative_to(RAIZ)))

    conf = {m: m7.get((m, CONFIRMATORIO), {}) for m in MODOS}
    primeira = {m: conf[m][min(conf[m])] for m in MODOS if conf[m]}

    # training: the first confirmatory seed per mode
    fazer("treino", lambda: graficos.treino({m: ler_jsonl(p / "passos.jsonl") for m, p in primeira.items()}) if primeira else None)
    sent = [l for m in MODOS for p in conf[m].values() for l in ler_jsonl(p / "sentinela.jsonl")]
    fazer("sentinela", lambda: graficos.sentinela(sent) if sent else None)

    # validation across the grid, and training against validation
    grade = {m: {ep: [ler_json(p / "avaliacao.json")["bpb"] for p in sem.values()] for (mm, ep), sem in m7.items() if mm == m} for m in MODOS}
    fazer("validacao_epocas", lambda: graficos.validacao_epocas(grade) if any(grade.values()) else None)
    curvas = {}
    for m in MODOS:
        pts = {"treino": [], "validacao": []}
        for (mm, ep), sem in sorted(m7.items()):
            tr = [ler_json(p / "avaliacao_treino.json") for p in sem.values()]
            if mm != m or any(x is None for x in tr):
                continue
            pts["treino"].append((ep, float(np.mean([x["bpb"] for x in tr]))))
            pts["validacao"].append((ep, float(np.mean([ler_json(p / "avaliacao.json")["bpb"] for p in sem.values()]))))
        if pts["treino"]:
            curvas[m] = pts
    resumo["treino_validacao"] = curvas
    fazer("treino_validacao", lambda: graficos.treino_validacao(curvas) if curvas else None)

    # test partition at the confirmatory point: Delta per work and with or without names
    sementes = sorted(set(conf["autoregressivo"]) & set(conf["difusao"]))
    teste = {m: [ler_json(conf[m][s] / "avaliacao_teste.json") for s in sementes] for m in MODOS}
    if sementes and all(x is not None for m in MODOS for x in teste[m]):
        d = est.dados_pareados(teste["difusao"], teste["autoregressivo"])
        b = est.bootstrap_pareado(d)
        por_obra = {}
        for g in sorted(set(d.grupos)):
            k = d.grupos == g
            dl = (d.nats_dif[:, k].sum(1) - d.nats_ar[:, k].sum(1)) / (np.log(2) * d.bytes_[k].sum())
            por_obra[g] = intervalo_t(dl)
        resumo["teste"] = {"sementes": sementes, "delta": b.estimativa, "ic95_bootstrap": [b.inferior, b.superior],
                           "p_aleatorizacao": est.teste_aleatorizacao(b.deltas_sementes),
                           "deltas_sementes": b.deltas_sementes.tolist(), "por_obra": por_obra,
                           "bpb": {m: [x["bpb"] for x in teste[m]] for m in MODOS}}
        fazer("delta_por_obra", lambda: graficos.delta_por_obra(por_obra, (b.estimativa, b.inferior, b.superior)))
        if "bpb_sem_nomes" in teste["difusao"][0]:
            ds = est.dados_pareados(teste["difusao"], teste["autoregressivo"], sem_nomes=True)
            bs = est.bootstrap_pareado(ds)
            resumo["teste"]["sem_nomes"] = {"delta": bs.estimativa, "ic95_bootstrap": [bs.inferior, bs.superior],
                                            "bpb": {m: [x["bpb_sem_nomes"] for x in teste[m]] for m in MODOS},
                                            "fracao_bytes_em_nomes": teste["difusao"][0]["fracao_bytes_em_nomes"]}
            fazer("com_sem_nomes", lambda: graficos.com_sem_nomes(
                {m: (float(np.mean([x["bpb"] for x in teste[m]])), float(np.mean([x["bpb_sem_nomes"] for x in teste[m]]))) for m in MODOS}))
    else:
        puladas += ["delta_por_obra (sem avaliacao_teste.json)", "com_sem_nomes (sem avaliacao_teste.json)"]

    # M9 samples: n-gram novelty and copy filter along the grid (corpus A only)
    amostras, controles = ler_jsonl(RAIZ / "leitura" / "amostras.jsonl"), ler_json(RAIZ / "leitura" / "controles.json")
    if corpus == "A" and amostras and controles:
        treino = [x["texto"] for x in ler_documentos("A", "treino")]
        humanos = [p[0] for p in controles["pares"]]
        ultimo = max(r["ponto"] for r in amostras)
        nov = {graficos.ROTULO[m]: novidade([r["texto"] for r in amostras if r["modo"] == m and r["ponto"] == ultimo], treino) for m in MODOS}
        nov["texto humano"] = novidade(humanos, treino)
        resumo["novidade_ngramas"] = {"ponto": ultimo, **nov}
        fazer("novidade_ngramas", lambda: graficos.novidade_ngramas(nov))
        desc = {}
        for m in MODOS:
            for ep in sorted({r["ponto"] for r in amostras}):
                c = [r["c"] for r in amostras if r["modo"] == m and r["ponto"] == ep]
                x = int(sum(v > controles["k"] for v in c))
                lo, hi = est.clopper_pearson(x, len(c))
                desc.setdefault(m, {})[ep] = (x / len(c), float(lo), float(hi))
        resumo["descarte_copia"] = {"k": controles["k"], **{m: {str(e): v for e, v in r.items()} for m, r in desc.items()}}
        resumo["n1"] = {m: {str(ep): float(np.mean([r["taxa_lexico"] for r in amostras if r["modo"] == m and r["ponto"] == ep]))
                            for ep in sorted({r["ponto"] for r in amostras})} for m in MODOS}
        resumo["n1"]["humano"] = float(np.mean(controles["taxas_humanos"]))
        fazer("descarte_copia", lambda: graficos.descarte_copia(desc))
    else:
        puladas += ["novidade_ngramas (sem amostras do M9)", "descarte_copia (sem amostras do M9)"]

    eg = ler_json(RAIZ / "resultados" / "entropia_genppl.json") if corpus == "A" else None
    fazer("entropia_genppl", lambda: graficos.entropia_genppl(
        {f"{graficos.ROTULO[k.split('_e')[0]]}, {k.split('_e')[1]} épocas": [(p["temperatura"], p["entropia"], p["genppl"]) for p in v]
         for k, v in eg.items()}) if eg else None)

    leitura = ler_json(RAIZ / "leitura" / "resultado.json")
    fazer("niveis", lambda: graficos.niveis(leitura["niveis"]) if leitura else None)

    # calibration and deletion: every run summarised, the first confirmatory seed drawn
    cal = collections.defaultdict(dict)
    for (m, ep), sem in sorted(m7.items()):
        cs = [ler_json(p / "calibracao.json") for p in sem.values()]
        ds = [ler_json(p / "delecao.json") for p in sem.values()]
        if any(x is None for x in cs + ds):
            continue
        cal[m][str(ep)] = {
            "T": intervalo_t([c["T"] for c in cs]), "ece_T1": intervalo_t([c["ece_T1"] for c in cs]),
            "ece_calibrado": intervalo_t([c["ece_calibrado"] for c in cs]), "auc_roc": intervalo_t([c["auc_roc"] for c in cs]),
            "portao_beta": {alvo: {"cobertura": intervalo_t([c["portao_beta"][alvo]["cobertura"] for c in cs]),
                                   "promessa_cumprida": sum(bool(c["portao_beta"][alvo]["promessa_cumprida"]) for c in cs), "n": len(cs)}
                            for alvo in cs[0]["portao_beta"]},
            "auc_delecao_importancia": intervalo_t([x["auc_importancia_media"] for x in ds]),
            "auc_delecao_aleatoria": intervalo_t([x["auc_aleatoria_media"] for x in ds]),
            "p_vazio_relativo": intervalo_t([x["p_vazio_relativo_medio"] for x in ds]),
            "spearman_p0_auc": intervalo_t([x["spearman_p0_auc"]["rho"] for x in ds])}
    resumo["calibracao_delecao"] = cal
    cfig = {p: (ler_json(p / "calibracao.json"), ler_json(p / "delecao.json")) for p in primeira.values()}
    confs, sels, dels = {}, {}, {}
    for m, p in primeira.items():
        c, dl = cfig[p]
        if c:
            for chave, rot in (("T1", "T=1"), ("calibrado", "calibrado")):
                confs[f"{m} {rot}"] = dict(c["confiabilidade"][chave], ece=c["ece_T1" if chave == "T1" else "ece_calibrado"])
            sels[m] = c["seletiva"]
        if dl:
            dels[m] = dl
    fazer("confiabilidade", lambda: graficos.confiabilidade(confs) if confs else None)
    fazer("seletiva", lambda: graficos.seletiva(sels) if sels else None)
    fazer("delecao", lambda: graficos.delecao(dels) if dels else None)

    # sensitivity: the M6 grid at the confirmatory point and the one-factor lines (corpus A)
    hp = collections.defaultdict(list)
    for p in (RAIZ / "runs").iterdir():
        h = NOME_HP.match(p.name)
        if h and (h.group(1) is not None) == (corpus == "B") and int(h.group(5)) == CONFIRMATORIO:
            av = ler_json(p / "avaliacao.json")
            if av:
                hp[(h.group(2), float(h.group(4)), float(h.group(3)))].append(av["bpb"])
    if hp:
        lrs = sorted({k[2] for k in hp})
        dos = sorted({k[1] for k in hp})
        g = {graficos.ROTULO[m]: [[float(np.mean(hp[(m, do, lr)])) if (m, do, lr) in hp else np.nan for lr in lrs] for do in dos] for m in MODOS}
        fazer("sensibilidade_grade", lambda: graficos.sensibilidade_grade(g, [f"{x:g}" for x in dos], [f"{x:g}" for x in lrs], "dropout", "taxa de aprendizado"))
    sens = ler_json(RAIZ / "resultados" / "sensibilidade.json") if corpus == "A" else None
    if sens:
        rot = {"L": ("L (tokens por janela)", True), "V": ("V (vocabulário)", True), "precisao": ("precisão de treino", False)}
        for param, (nome, xlog) in rot.items():
            linhas = sens["fatores"].get(param, {})
            if len(linhas) < 2:
                continue
            chave = (lambda v: float(v)) if param != "precisao" else (lambda v: {"fp16": 16.0, "fp32": 32.0}[v])
            dados = {graficos.ROTULO[m]: {chave(v): r[f"bpb_{m}"]["por_semente"] for v, r in linhas.items()} for m in MODOS}
            fazer(f"sensibilidade_{param}", lambda d=dados, n=nome, x=xlog: graficos.sensibilidade_linha(d, n, "bits por byte (validação)", xlog=x))
        k = sens["fatores"].get("K")
        if k:
            fazer("sensibilidade_K", lambda: graficos.sensibilidade_linha(
                {"difusão": {float(v): r["bpb_difusao"]["por_semente"] for v, r in k.items()}},
                "K (sorteios da NELBO)", "bits por byte (validação)", xlog=True))
        tg = sens["fatores"].get("T_gen")
        if tg:
            fazer("sensibilidade_T_gen", lambda: graficos.sensibilidade_linha(
                {"difusão": {float(v): r["taxa_lexico"]["por_semente"] for v, r in tg.items() if v.isdigit()}},
                "T_gen (passos de geração)", "taxa de léxico das amostras", xlog=True, melhor="alto",
                referencia=(tg["humanos"]["taxa_lexico"], "texto humano de teste")))

    (RAIZ / "resultados" / f"m7_analises{sufixo}.json").write_text(json.dumps(resumo, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps({"feitas": feitas, "puladas": puladas}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Milestone M6: hyperparameter search, frozen Delta_min and pilot (docs/decisao_orcamento.md).

Subcommands, run in order; the runs ones stop after --max-min minutes and resume where they left:
    preregistrar   write preregistro/m6_busca.json with its sha256 (refuses to overwrite)
    busca          train the pending runs of the pre-registered grid
    selecionar     choose one configuration per mode by the pre-registered rule, compute Delta_min
                   and freeze them in preregistro/m6_delta_min.json
    piloto         train the 10 pilot seeds per mode at 256 epochs with the chosen configurations
    sementes       pilot sigma, number of confirmatory seeds and bootstrap coverage

    python scripts/m6.py busca --max-min 60
    python scripts/m6.py --corpus B preregistrar      # same protocol on corpus B, separate names
"""
import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from slm.corpus import hash_corpus  # noqa: E402
from slm.registro import conferir_preregistro, preregistrar  # noqa: E402

PRE = RAIZ / "preregistro"
RUNS = RAIZ / "runs"
GRADE = {"lr": [3e-4, 1e-3, 3e-3], "dropout": [0.0, 0.1, 0.2], "epocas": [16, 256], "sementes": [0, 1],
         "modos": ["autoregressivo", "difusao"]}
PILOTO = {"epocas": 256, "sementes": list(range(100, 110))}
CORPUS = "A"


def suf(sep: str = "-") -> str:
    """Name suffix of the corpus: empty for A, which keeps the names already registered."""
    return "" if CORPUS == "A" else f"{sep}{CORPUS}"


def run_id_hp(modo, lr, do, ep, s):
    return f"hp{suf()}-{modo}-lr{lr:g}-do{do:g}-e{ep}-s{s}"


def bpb(run_id):
    p = RUNS / run_id / "avaliacao.json"
    return json.loads(p.read_text(encoding="utf-8"))["bpb"] if p.exists() else None


def treinar(run_id, modo, ep, s, lr, do, papel):
    cmd = [sys.executable, str(RAIZ / "scripts" / "treinar.py"), "--modo", modo, "--epocas", str(ep), "--semente", str(s),
           "--lr", str(lr), "--dropout", str(do), "--papel", papel, "--precisao", "fp16", "--atencao", "efficient",
           "--dispositivo", "cuda", "--corpus", CORPUS, "--run-id", run_id]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        sys.exit(f"{run_id} failed:\n{r.stderr[-2000:]}")
    print(run_id, round(json.loads(r.stdout)["bpb_validacao"], 4), flush=True)


def lote(tarefas, max_min):
    t0 = time.time()
    for t in tarefas:
        if (RUNS / t[0] / "avaliacao.json").exists():
            continue
        if (time.time() - t0) / 60 > max_min:
            print("time budget reached, rerun to resume")
            return
        treinar(*t)
    print("all runs of this stage done")


def cmd_preregistrar(_):
    h = preregistrar(PRE, f"m6_busca{suf('_')}", {
        "grade": GRADE, "fixos": {"lote": 64, "weight_decay": 0.1, "aquecimento_frac": 0.02, "lr_min_frac": 0.1,
                                  "L": 256, "rede": "6x384", "precisao": "fp16", "atencao": "efficient"},
        "corpus": CORPUS, "corpus_hash": hash_corpus(CORPUS),
        "tokenizador_sha256": json.loads((RAIZ / "tokenizador" / CORPUS / "metadados.json").read_text())["sha256_tokenizer_json"],
        "metrica": "bpb de validacao, razao de somas; difusao pela NELBO com K=1 amostra por janela",
        "selecao": ("por modo, a configuracao de menor media de bpb entre as 2 sementes no ponto de 256 epocas; "
                    "sigma_hat = raiz da media, sobre as 9 configuracoes, da variancia entre as 2 sementes, por modo e ponto; "
                    "se as duas melhores diferirem menos que sigma_hat, vence a de maior dropout e, empatando, a de menor lr"),
        "delta_min": "1/4 de (bpb_AR com 4 epocas - bpb_AR com 16 epocas), na configuracao AR escolhida, media de 2 sementes",
        "piloto": PILOTO})
    print(f"m6_busca{suf('_')}", h)


def tarefas_busca():
    for modo, lr, do, ep, s in itertools.product(GRADE["modos"], GRADE["lr"], GRADE["dropout"], GRADE["epocas"], GRADE["sementes"]):
        yield (run_id_hp(modo, lr, do, ep, s), modo, ep, s, lr, do, "hp")


def cmd_busca(a):
    assert conferir_preregistro(PRE, f"m6_busca{suf('_')}"), "pre-registration missing or altered"
    lote(list(tarefas_busca()), a.max_min)


def cmd_selecionar(_):
    assert conferir_preregistro(PRE, f"m6_busca{suf('_')}")
    escolha, tabela = {}, {}
    for modo in GRADE["modos"]:
        cfgs = list(itertools.product(GRADE["lr"], GRADE["dropout"]))
        v = {c: [bpb(run_id_hp(modo, *c, 256, s)) for s in GRADE["sementes"]] for c in cfgs}
        if any(x is None for xs in v.values() for x in xs):
            sys.exit(f"search incomplete for {modo}")
        sigma = float(np.sqrt(np.mean([np.var(xs, ddof=1) for xs in v.values()])))
        ordem = sorted(cfgs, key=lambda c: np.mean(v[c]))
        a, b = ordem[0], ordem[1]
        if np.mean(v[b]) - np.mean(v[a]) < sigma and (b[1], -b[0]) > (a[1], -a[0]):
            a = b
        escolha[modo] = {"lr": a[0], "dropout": a[1]}
        tabela[modo] = {"sigma_hat": sigma, "bpb_256": {f"lr{c[0]:g}-do{c[1]:g}": v[c] for c in cfgs}}
    lr, do = escolha["autoregressivo"]["lr"], escolha["autoregressivo"]["dropout"]
    for ep in (4,):
        for s in GRADE["sementes"]:
            rid = run_id_hp("autoregressivo", lr, do, ep, s)
            if not (RUNS / rid / "avaliacao.json").exists():
                treinar(rid, "autoregressivo", ep, s, lr, do, "hp")
    b4 = [bpb(run_id_hp("autoregressivo", lr, do, 4, s)) for s in GRADE["sementes"]]
    b16 = [bpb(run_id_hp("autoregressivo", lr, do, 16, s)) for s in GRADE["sementes"]]
    ganhos = np.subtract(b4, b16)
    delta_min = float(np.mean(ganhos) / 4)
    h = preregistrar(PRE, f"m6_delta_min{suf('_')}", {
        "escolha": escolha, "busca": tabela, "bpb_AR_4": b4, "bpb_AR_16": b16,
        "delta_min": delta_min, "erro_padrao_delta_min": float(np.std(ganhos, ddof=1) / np.sqrt(len(ganhos)) / 4),
        "m_prec": delta_min / 4, "valido": bool(delta_min > 0)})
    print(json.dumps({"escolha": escolha, "delta_min": delta_min, "hash": h}, indent=2))


def cmd_piloto(a):
    assert conferir_preregistro(PRE, f"m6_delta_min{suf('_')}"), "freeze Delta_min first"
    esc = json.loads((PRE / f"m6_delta_min{suf('_')}.json").read_text())["escolha"]
    t = [(f"piloto{suf()}-{m}-e{PILOTO['epocas']}-s{s}", m, PILOTO["epocas"], s, esc[m]["lr"], esc[m]["dropout"], "piloto")
         for s in PILOTO["sementes"] for m in GRADE["modos"]]
    lote(t, a.max_min)


def cmd_sementes(_):
    from slm.estatistica import numero_sementes, sigma_sup  # noqa: E402
    pre = json.loads((PRE / f"m6_delta_min{suf('_')}.json").read_text())
    d = [bpb(f"piloto{suf()}-difusao-e256-s{s}") - bpb(f"piloto{suf()}-autoregressivo-e256-s{s}") for s in PILOTO["sementes"]]
    sig = float(np.std(d, ddof=1))
    print(json.dumps({"deltas_piloto": d, "sigma_hat": sig, "sigma_sup": sigma_sup(sig, len(d)),
                      "n": numero_sementes(sig, pre["delta_min"], m=len(d))}, indent=2, default=str))


def main():
    global CORPUS
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["preregistrar", "busca", "selecionar", "piloto", "sementes"])
    ap.add_argument("--max-min", type=float, default=60)
    ap.add_argument("--corpus", default="A")
    a = ap.parse_args()
    CORPUS = a.corpus
    {"preregistrar": cmd_preregistrar, "busca": cmd_busca, "selecionar": cmd_selecionar,
     "piloto": cmd_piloto, "sementes": cmd_sementes}[a.etapa](a)


if __name__ == "__main__":
    main()

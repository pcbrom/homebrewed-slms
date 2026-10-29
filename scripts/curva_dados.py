#!/usr/bin/env python3
"""Data curve: from how much text does the Transformer beat the n-gram floor? (decisao_avaliacao.md)

The autoregressive Transformer, with the hyperparameters of the corpus chosen in M6, and the
Kneser-Ney n-gram are trained on nested fractions of the training groups (1/8, 1/4, 1/2, all).
Per fraction, the Transformer runs at 16 and 64 epochs with 3 seeds and the budget with the lower
mean validation bits per byte is kept; the n-gram order is chosen on validation among 3 to 6. Both,
and their mixture with the weight fitted per seed on validation, are then scored on the external
laws of 2025 and 2026, which no model has seen.

    python scripts/curva_dados.py --corpus B preregistrar
    python scripts/curva_dados.py --corpus B rodar --max-min 60
    python scripts/curva_dados.py --corpus B analisar
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from slm.avaliacao import logp_tokens_ar  # noqa: E402
from slm.corpus import fracao_grupos, ler_documentos, tokenizar  # noqa: E402
from slm.modelo import ConfigRede, Transformer  # noqa: E402
from slm.ngrama import NGramaKN  # noqa: E402
from slm.registro import conferir_preregistro, preregistrar  # noqa: E402
from slm.tokenizador import carregar_tokenizador  # noqa: E402

PRE, RUNS = RAIZ / "preregistro", RAIZ / "runs"
FRACOES = (0.125, 0.25, 0.5, 1.0)
ORCAMENTOS = (16, 64)
SEMENTES = (500, 501, 502)
ORDENS = (3, 4, 5, 6)
L = 256
LAMBDAS = np.linspace(0, 1, 101)


def nome(corpus):
    return f"curva_dados_{corpus}"


def run_id(corpus, fr, ep, s):
    return f"curva-{corpus}-autoregressivo-f{fr:g}-e{ep}-s{s}"


def cmd_preregistrar(a):
    m6 = f"m6_delta_min_{a.corpus}" if a.corpus != "A" else "m6_delta_min"
    assert conferir_preregistro(PRE, m6), "M6 of the corpus not frozen"
    d = json.loads((PRE / f"{m6}.json").read_text(encoding="utf-8"))
    h = preregistrar(PRE, nome(a.corpus), {
        "corpus": a.corpus, "modo": "autoregressivo", "hiperparametros": d["escolha"]["autoregressivo"],
        "fracoes": list(FRACOES), "selecao_fracao": "prefixo de grupos inteiros em ordem sha256(semente:grupo), aninhado",
        "orcamentos": list(ORCAMENTOS), "sementes": list(SEMENTES),
        "escolha_orcamento": "menor media de bits por byte na validacao entre os orcamentos, por fracao",
        "ngrama": {"ordens": list(ORDENS), "escolha": "menor bits por byte na validacao, por fracao"},
        "mistura": "p = lam p_T + (1 - lam) p_ng, lam em passos de 0,01 ajustado na validacao por semente",
        "avaliacao": "conjunto externo B_externo, mesmas janelas de 256, preenchimento fora",
        "delta_min": d["delta_min"],
        "leitura": ("por fracao, D = bpb_ngrama - bpb_T no externo, media das sementes com IC de 95% por t com 2 gl; "
                    "o Transformer vale a pena na fracao quando o limite inferior do IC passa delta_min; "
                    "a mistura e relatada com o mesmo intervalo, sem criterio"),
    })
    print(nome(a.corpus), h)


def cmd_rodar(a):
    assert conferir_preregistro(PRE, nome(a.corpus)), "pre-registration missing or altered"
    pre = json.loads((PRE / f"{nome(a.corpus)}.json").read_text(encoding="utf-8"))
    hp, t0 = pre["hiperparametros"], time.time()
    for fr in pre["fracoes"]:
        for ep in pre["orcamentos"]:
            for s in pre["sementes"]:
                rid = run_id(a.corpus, fr, ep, s)
                if (RUNS / rid / "avaliacao.json").exists():
                    continue
                if (time.time() - t0) / 60 > a.max_min:
                    print("time budget reached, rerun to resume")
                    return
                cmd = [sys.executable, str(RAIZ / "scripts" / "treinar.py"), "--modo", "autoregressivo",
                       "--epocas", str(ep), "--semente", str(s), "--lr", str(hp["lr"]), "--dropout", str(hp["dropout"]),
                       "--papel", "curva", "--precisao", "fp16", "--atencao", "efficient", "--dispositivo", "cuda",
                       "--corpus", a.corpus, "--fracao-treino", str(fr), "--run-id", rid]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
                if r.returncode != 0:
                    sys.exit(f"{rid} falhou:\n{r.stderr[-2000:]}")
                print(rid, round(json.loads(r.stdout)["bpb_validacao"], 4), flush=True)
    print("all data-curve runs done")


def bpb_val(rid):
    return json.loads((RUNS / rid / "avaliacao.json").read_text(encoding="utf-8"))["bpb"]


def carregar(rid, dispositivo):
    cfg = json.loads((RUNS / rid / "config.json").read_text(encoding="utf-8"))
    m = Transformer(ConfigRede(vocab_size=cfg["vocab_size"], L=cfg["L"], n_layer=cfg["n_layer"], d=cfg["d"],
                               n_head=cfg["n_head"]))
    m.load_state_dict(torch.load(RUNS / rid / "pesos.pt", map_location="cpu"))
    return m.to(dispositivo).eval()


def cmd_analisar(a):
    pre = json.loads((PRE / f"{nome(a.corpus)}.json").read_text(encoding="utf-8"))
    disp = "cuda" if torch.cuda.is_available() else "cpu"
    tok = carregar_tokenizador(RAIZ / "tokenizador" / a.corpus)
    treino_todo = ler_documentos(a.corpus, "treino")
    val = tokenizar(ler_documentos(a.corpus, "validacao"), tok)
    ext = tokenizar(ler_documentos(f"{a.corpus}_externo"), tok)
    b_val, b_ext = sum(d["bytes"] for d in val), sum(d["bytes"] for d in ext)
    bpb = lambda lp, b: float(-lp.sum() / (math.log(2) * b))
    saida = {"corpus": a.corpus, "delta_min": pre["delta_min"], "fracoes": {}}
    for fr in pre["fracoes"]:
        sub = fracao_grupos(treino_todo, fr)
        ids = [d["ids"] for d in tokenizar(sub, tok)]
        ngs = {n: NGramaKN(n, tok.vocab_size).treinar(ids) for n in pre["ngrama"]["ordens"]}
        ng_val = {n: np.concatenate([m.logp_janelas(d["ids"], L) for d in val]) for n, m in ngs.items()}
        n_esc = min(ng_val, key=lambda n: -ng_val[n].sum())
        lp_ng_val = ng_val[n_esc]
        lp_ng_ext = np.concatenate([ngs[n_esc].logp_janelas(d["ids"], L) for d in ext])
        medias = {ep: np.mean([bpb_val(run_id(a.corpus, fr, ep, s)) for s in pre["sementes"]]) for ep in pre["orcamentos"]}
        ep = min(medias, key=medias.get)
        por_semente = []
        for s in pre["sementes"]:
            m = carregar(run_id(a.corpus, fr, ep, s), disp)
            lp_t_val = np.concatenate(logp_tokens_ar(m, [d["ids"] for d in val], L, dispositivo=disp))
            lp_t_ext = np.concatenate(logp_tokens_ar(m, [d["ids"] for d in ext], L, dispositivo=disp))
            mist = lambda lam, t, g: np.logaddexp(np.log(lam + 1e-300) + t, np.log(1 - lam + 1e-300) + g)
            lam = float(LAMBDAS[np.argmax([mist(l, lp_t_val, lp_ng_val).sum() for l in LAMBDAS])])
            por_semente.append({"semente": s, "lambda": lam, "bpb_transformer_externo": bpb(lp_t_ext, b_ext),
                                "bpb_mistura_externo": bpb(mist(lam, lp_t_ext, lp_ng_ext), b_ext)})
        bpb_ng_ext = bpb(lp_ng_ext, b_ext)

        def intervalo(v):
            v = np.asarray(v)
            h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v))
            return [float(v.mean() - h), float(v.mean() + h)]

        d_t = [bpb_ng_ext - r["bpb_transformer_externo"] for r in por_semente]
        d_m = [bpb_ng_ext - r["bpb_mistura_externo"] for r in por_semente]
        saida["fracoes"][str(fr)] = {
            "bytes_treino": sum(d["bytes"] for d in sub), "grupos_treino": len({d["grupo"] for d in sub}),
            "ordem_ngrama": n_esc, "bpb_ngrama_validacao": bpb(lp_ng_val, b_val), "bpb_ngrama_externo": bpb_ng_ext,
            "orcamento": ep, "bpb_transformer_validacao_por_orcamento": medias, "sementes": por_semente,
            "ganho_transformer_sobre_ngrama": {"media": float(np.mean(d_t)), "ic95": intervalo(d_t)},
            "ganho_mistura_sobre_ngrama": {"media": float(np.mean(d_m)), "ic95": intervalo(d_m)},
            "transformer_vale_a_pena": bool(intervalo(d_t)[0] > pre["delta_min"]),
        }
        r = saida["fracoes"][str(fr)]
        print(fr, r["bytes_treino"], "ngrama", n_esc, round(bpb_ng_ext, 4), "T", ep,
              round(np.mean([x["bpb_transformer_externo"] for x in por_semente]), 4),
              "mistura", round(np.mean([x["bpb_mistura_externo"] for x in por_semente]), 4),
              "vale", r["transformer_vale_a_pena"], flush=True)
    (RAIZ / "resultados" / f"{nome(a.corpus)}.json").write_text(json.dumps(saida, indent=1, ensure_ascii=False),
                                                                 encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="B")
    sub = ap.add_subparsers(dest="etapa", required=True)
    sub.add_parser("preregistrar")
    s = sub.add_parser("rodar")
    s.add_argument("--max-min", type=float, default=60)
    sub.add_parser("analisar")
    a = ap.parse_args()
    {"preregistrar": cmd_preregistrar, "rodar": cmd_rodar, "analisar": cmd_analisar}[a.etapa](a)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Milestone M7: descriptive grid and confirmatory point (docs/decisao_orcamento.md).

Subcommands, run in order:
    preregistrar  bootstrap coverage on the pilot (sign-flip nulls), then freeze points, seeds,
                  hyperparameters and the reading rule in preregistro/m7.json, and write the run list
                  read by the blind reading in preregistro/m7_grade.json
    rodar         train the pending runs (descriptive points without sentinel, confirmatory point
                  with the precision sentinel until the end); stops after --max-min and resumes
    analisar      two-level bootstrap interval of Delta = bpb_dif - bpb_AR at every point and the
                  pre-registered reading at the confirmatory point; writes resultados/m7.json

    python scripts/m7.py preregistrar
    python scripts/m7.py rodar --max-min 60
    python scripts/m7.py analisar
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from slm import estatistica as est  # noqa: E402
from slm.registro import conferir_preregistro, preregistrar  # noqa: E402

PRE, RUNS = RAIZ / "preregistro", RAIZ / "runs"
MODOS = ("autoregressivo", "difusao")
DESCRITIVOS = (1, 4, 16, 64)
SEMENTES_DESC = (200, 201, 202, 203, 204)
CONFIRMATORIO = 256
PILOTO = tuple(range(100, 110))
CORPUS = "A"


def suf(sep: str = "-") -> str:
    """Name suffix of the corpus: empty for A, which keeps the names already registered."""
    return "" if CORPUS == "A" else f"{sep}{CORPUS}"


def run_id(modo, ep, s, papel):
    return f"piloto{suf()}-{modo}-e{ep}-s{s}" if papel == "piloto" else f"m7{suf()}-{modo}-e{ep}-s{s}"


def avaliacao(rid):
    p = RUNS / rid / "avaliacao.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pareados(ep, sementes, papel):
    dif = [avaliacao(run_id("difusao", ep, s, papel)) for s in sementes]
    ar = [avaliacao(run_id("autoregressivo", ep, s, papel)) for s in sementes]
    if any(x is None for x in dif + ar):
        return None
    return est.dados_pareados(dif, ar)


def cmd_preregistrar(a):
    assert conferir_preregistro(PRE, f"m6_delta_min{suf('_')}"), "M6 not frozen"
    m6 = json.loads((PRE / f"m6_delta_min{suf('_')}.json").read_text(encoding="utf-8"))
    d = pareados(CONFIRMATORIO, PILOTO, "piloto")
    cob = est.cobertura_troca_sinal(d.nats_dif, d.nats_ar, d.bytes_, d.grupos, n_nulos=a.nulos, R=a.R)
    deltas = (d.nats_dif.sum(1) - d.nats_ar.sum(1)) / (np.log(2) * d.bytes_.sum())
    n = est.numero_sementes(est.desvio_piloto(deltas), m6["delta_min"], m=len(PILOTO))
    n_conf = n.n or n.teto
    if CORPUS != "A" and not cob.aprovada:
        # rule of the corpus A amendment, pre-registered here from the start: the randomization
        # fallback needs at least 6 seeds to be able to reject at 0.05
        n_conf = max(n_conf, 6)
    sem_conf = tuple(range(300, 300 + n_conf))
    h = preregistrar(PRE, f"m7{suf('_')}", {
        "corpus": CORPUS,
        "escolha": m6["escolha"], "delta_min": m6["delta_min"],
        "pontos_descritivos": DESCRITIVOS, "sementes_descritivas": SEMENTES_DESC,
        "ponto_confirmatorio": CONFIRMATORIO, "sementes_confirmatorias": sem_conf,
        "numero_sementes": {"n": n.n, "resolvivel": n.resolvivel, "sigma_sup": n.sigma_sup, "teto": n.teto},
        "cobertura_bootstrap_piloto": {"cobertura": cob.cobertura, "n_nulos": cob.n_nulos, "R": a.R, "aprovada": cob.aprovada},
        "inferencia": ("bootstrap em dois niveis (obras e sementes), R = 10.000, intervalo de 95%" if cob.aprovada else
                       "teste de aleatorizacao sobre as sementes, afirmacao restrita ao conjunto de avaliacao"),
        "leitura": "Delta = bpb_dif - bpb_AR na validacao; abaixo de zero: difusao melhor (conclusivo); acima: AR supera o limite da difusao (sem conclusao sobre a difusao); contem zero: inconclusivo",
        "sentinela": "ativa ate o fim no ponto confirmatorio (precisao G4)",
        "precisao": "fp16 com GradScaler, atencao efficient"})
    grade = [{"run_id": run_id(m, ep, s, "m7"), "modo": m, "ponto": ep, "semente": s}
             for ep in DESCRITIVOS for s in SEMENTES_DESC[:4] for m in MODOS]
    grade += [{"run_id": run_id(m, CONFIRMATORIO, s, "piloto"), "modo": m, "ponto": CONFIRMATORIO, "semente": s}
              for s in PILOTO[:4] for m in MODOS]
    (PRE / f"m7_grade{suf('_')}.json").write_text(json.dumps(grade, indent=1), encoding="utf-8")
    print(json.dumps({"hash": h, "cobertura": cob.cobertura, "aprovada": cob.aprovada, "sementes_confirmatorias": sem_conf}, indent=2))


def cmd_emendar(a):
    """Amendment frozen before any confirmatory run: more seeds so the randomization fallback can reject."""
    assert conferir_preregistro(PRE, f"m7{suf('_')}"), "M7 not frozen"
    pre = json.loads((PRE / f"m7{suf('_')}.json").read_text(encoding="utf-8"))
    if any((RUNS / run_id(m, pre["ponto_confirmatorio"], s, "m7")).exists() for s in range(300, 300 + a.sementes) for m in MODOS):
        sys.exit("confirmatory runs already exist: an amendment after data is not allowed here")
    h = preregistrar(PRE, f"m7_emenda{suf('_')}", {
        "data": "2026-10-27", "aprovada_por": "autor", "antes_de_qualquer_corrida_confirmatoria": True,
        "motivo": ("a cobertura do bootstrap no piloto ficou abaixo de 0,93, e o pre-registro passa a inferencia ao teste de "
                   "aleatorizacao por troca de sinal sobre as sementes; com 3 sementes ha 8 trocas e o menor p bilateral e 0,25, "
                   "entao o teste nao rejeitaria a 0,05 com nenhum efeito"),
        "regra": "n = maior entre o da formula de poder e o menor n com 2 * 2^-n < 0,05, isto e, 6",
        "sementes_confirmatorias": tuple(range(300, 300 + a.sementes))})
    print("m7_emenda", h)


def treinar(rid, modo, ep, s, hp, papel, sentinela):
    cmd = [sys.executable, str(RAIZ / "scripts" / "treinar.py"), "--modo", modo, "--epocas", str(ep), "--semente", str(s),
           "--lr", str(hp["lr"]), "--dropout", str(hp["dropout"]), "--papel", papel, "--precisao", "fp16",
           "--atencao", "efficient", "--dispositivo", "cuda", "--corpus", CORPUS, "--run-id", rid] + (["--sentinela"] if sentinela else [])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        sys.exit(f"{rid} falhou:\n{r.stderr[-2000:]}")
    print(rid, round(json.loads(r.stdout)["bpb_validacao"], 4), flush=True)


def sementes_confirmatorias(pre):
    """Confirmatory seeds: those of the amendment preregistro/m7_emenda.json when it exists."""
    if (PRE / f"m7_emenda{suf('_')}.json").exists():
        assert conferir_preregistro(PRE, f"m7_emenda{suf('_')}"), "amendment altered"
        return json.loads((PRE / f"m7_emenda{suf('_')}.json").read_text(encoding="utf-8"))["sementes_confirmatorias"]
    return pre["sementes_confirmatorias"]


def cmd_rodar(a):
    assert conferir_preregistro(PRE, f"m7{suf('_')}"), "pre-registration missing or altered"
    pre = json.loads((PRE / f"m7{suf('_')}.json").read_text(encoding="utf-8"))
    tarefas = [(ep, s, "descritivo", False) for ep in pre["pontos_descritivos"] for s in pre["sementes_descritivas"]]
    if not a.so_descritivos:
        tarefas += [(pre["ponto_confirmatorio"], s, "confirmatoria", True) for s in sementes_confirmatorias(pre)]
    t0 = time.time()
    for ep, s, papel, sentinela in tarefas:
        for modo in MODOS:
            rid = run_id(modo, ep, s, "m7")
            if (RUNS / rid / "avaliacao.json").exists():
                continue
            if (time.time() - t0) / 60 > a.max_min:
                print("time budget reached, rerun to resume")
                return
            treinar(rid, modo, ep, s, pre["escolha"][modo], papel, sentinela)
    print("all M7 runs done")


def cmd_analisar(_):
    pre = json.loads((PRE / f"m7{suf('_')}.json").read_text(encoding="utf-8"))
    saida = {}
    pontos = [(ep, pre["sementes_descritivas"]) for ep in pre["pontos_descritivos"]] + [(pre["ponto_confirmatorio"], sementes_confirmatorias(pre))]
    for ep, sementes in pontos:
        d = pareados(ep, sementes, "m7")
        if d is None:
            continue
        b = est.bootstrap_pareado(d)
        bpb_dif = [est.bpb(n.sum(), d.bytes_.sum()) for n in d.nats_dif]
        bpb_ar = [est.bpb(n.sum(), d.bytes_.sum()) for n in d.nats_ar]
        r = {"sementes": list(sementes), "bpb_difusao": bpb_dif, "bpb_autoregressivo": bpb_ar,
             "delta": float(np.mean(bpb_dif) - np.mean(bpb_ar)), "ic95": [b.inferior, b.superior]}
        if ep == pre["ponto_confirmatorio"]:
            if pre["cobertura_bootstrap_piloto"]["aprovada"]:
                r["leitura"], r["texto"] = est.ler_resultado(b.inferior, b.superior)
            else:
                # pre-registered fallback: sign-flip randomization over seeds, claim restricted to the evaluation set
                r["p_aleatorizacao"] = est.teste_aleatorizacao(b.deltas_sementes)
                if r["p_aleatorizacao"] < 0.05:
                    r["leitura"], r["texto"] = est.ler_resultado(*((-1.0, -0.5) if r["delta"] < 0 else (0.5, 1.0)))
                else:
                    r["leitura"], r["texto"] = est.ler_resultado(-1.0, 1.0)
            r["acima_delta_min_em_modulo"] = bool(abs(r["delta"]) >= pre["delta_min"])
        saida[str(ep)] = r
        print(ep, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("bpb_difusao", "bpb_autoregressivo", "texto")})
    (RAIZ / "resultados").mkdir(exist_ok=True)
    (RAIZ / "resultados" / f"m7{suf('_')}.json").write_text(json.dumps(saida, indent=1, ensure_ascii=False), encoding="utf-8")


def main():
    global CORPUS
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="A")
    sub = ap.add_subparsers(dest="etapa", required=True)
    s = sub.add_parser("emendar")
    s.add_argument("--sementes", type=int, required=True)
    s = sub.add_parser("preregistrar")
    s.add_argument("--nulos", type=int, default=1000)
    s.add_argument("--R", type=int, default=10_000)
    s = sub.add_parser("rodar")
    s.add_argument("--max-min", type=float, default=60)
    s.add_argument("--so-descritivos", action="store_true")
    sub.add_parser("analisar")
    a = ap.parse_args()
    CORPUS = a.corpus
    {"preregistrar": cmd_preregistrar, "emendar": cmd_emendar, "rodar": cmd_rodar, "analisar": cmd_analisar}[a.etapa](a)


if __name__ == "__main__":
    main()

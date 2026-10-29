#!/usr/bin/env python3
"""Trained comparison of the Morse-style code against BPE at equal compute (docs/decisao_tokenizador_codigo.md).

Subcommands:
    plano     CPU only: symbols per byte of each arm on the training partition, training cost per
              symbol, epochs that equalise compute with B1 and the window that equalises context in
              bytes; writes resultados/m10/m10_plano.json
    rodar     GPU: trains the pending runs (both windows, both compute points, pilot seeds) with
              scripts/treinar.py; stops after --max-min minutes and resumes where it left
    analisar  bits per byte of each run and the paired difference against B1 per window, point and
              mode, with a t interval over seeds

    python experimentos/m10_codigo/m10_experimento.py plano
    python experimentos/m10_codigo/m10_experimento.py rodar --modo autoregressivo --max-min 60
    python experimentos/m10_codigo/m10_experimento.py analisar
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
SAIDA = RAIZ / "resultados" / "m10"
sys.path.insert(0, str(RAIZ))
from slm.corpus import ler_documentos  # noqa: E402
from slm.tokenizador import carregar_tokenizador  # noqa: E402

BRACOS = {"B1": RAIZ / "tokenizador" / "A", "B0": RAIZ / "tokenizador" / "m10_B0", "C1": RAIZ / "tokenizador" / "m10_C1",
          "C2": RAIZ / "tokenizador" / "m10_C2", "C3": RAIZ / "tokenizador" / "m10_C3"}
REDE = {"n_layer": 6, "d": 384, "n_head": 6}
L_BASE, PONTOS, SEMENTES = 256, (16, 256), (0, 1, 2)
PLANO = SAIDA / "m10_plano.json"


def custo(V: int, L: int) -> float:
    n, d = REDE["n_layer"], REDE["d"]
    return 6 * 12 * n * d * d + 6 * n * L * d + 6 * d * V


def cmd_plano(_):
    treino = [d["texto"] for d in ler_documentos("A", "treino")]
    B = sum(len(t.encode("utf-8")) for t in treino)
    bracos = {}
    for nome, pasta in BRACOS.items():
        tok = carregar_tokenizador(pasta)
        S = sum(len(tok.encode(t)) for t in treino)
        bracos[nome] = {"pasta": str(pasta.relative_to(RAIZ)), "V": tok.vocab_size, "simbolos_treino": S,
                        "simbolos_por_byte": S / B}
    ref = bracos["B1"]
    for nome, b in bracos.items():
        L_bytes = max(16, round(L_BASE * b["simbolos_por_byte"] / ref["simbolos_por_byte"]))
        b["janelas"] = {"simbolos": L_BASE, "bytes": L_bytes}
        b["contexto_bytes_janela_simbolos"] = L_BASE / b["simbolos_por_byte"]
        b["fator_epocas"] = {j: (ref["simbolos_treino"] * custo(ref["V"], L_BASE)) / (b["simbolos_treino"] * custo(b["V"], L))
                             for j, L in b["janelas"].items()}
    PLANO.write_text(json.dumps({"bytes_treino": B, "bracos": bracos}, indent=1), encoding="utf-8")
    print(f"{'braco':5s} {'V':>5s} {'simb/byte':>9s} {'contexto (bytes)':>16s} {'L iguala bytes':>14s} {'epocas x (simb)':>15s} {'epocas x (bytes)':>16s}")
    for nome, b in bracos.items():
        print(f"{nome:5s} {b['V']:5d} {b['simbolos_por_byte']:9.4f} {b['contexto_bytes_janela_simbolos']:16.0f} "
              f"{b['janelas']['bytes']:14d} {b['fator_epocas']['simbolos']:15.3f} {b['fator_epocas']['bytes']:16.3f}")


def run_id(braco, janela, ponto, semente, modo):
    return f"m10-{braco}-{janela}-e{ponto}-s{semente}-{modo[:3]}"


def hiperparametros():
    pre = RAIZ / "preregistro" / "m6_delta_min.json"
    if not pre.exists():
        sys.exit("hiperparametros do M6 ainda nao congelados: rode o M6 antes")
    esc = json.loads(pre.read_text(encoding="utf-8"))["escolha"]
    return esc


def cmd_rodar(a):
    plano = json.loads(PLANO.read_text(encoding="utf-8"))
    hp = hiperparametros()[a.modo]
    t0 = time.time()
    for ponto in PONTOS:
        for semente in SEMENTES:
            for nome, b in plano["bracos"].items():
                for janela, L in b["janelas"].items():
                    if janela == "bytes" and nome == "B1":
                        continue
                    rid = run_id(nome, janela, ponto, semente, a.modo)
                    if (RAIZ / "runs" / rid / "avaliacao.json").exists():
                        continue
                    if (time.time() - t0) / 60 > a.max_min:
                        print("orcamento de tempo atingido, rode de novo para continuar")
                        return
                    cfg = AQUI / "cfg" / f"{rid}.json"
                    cfg.parent.mkdir(exist_ok=True)
                    cfg.write_text(json.dumps({**REDE, "L": L, "corpus": "A"}), encoding="utf-8")
                    epocas = ponto * b["fator_epocas"][janela]
                    cmd = [sys.executable, str(RAIZ / "scripts" / "treinar.py"), "--config", str(cfg), "--modo", a.modo,
                           "--epocas", f"{epocas:.6f}", "--semente", str(semente), "--lr", str(hp["lr"]), "--dropout", str(hp["dropout"]),
                           "--papel", "tokenizador", "--precisao", "fp16", "--atencao", "efficient", "--dispositivo", "cuda",
                           "--tokenizador", str(RAIZ / b["pasta"]), "--run-id", rid]
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
                    if r.returncode != 0:
                        sys.exit(f"{rid} falhou:\n{r.stderr[-2000:]}")
                    print(rid, round(json.loads(r.stdout)["bpb_validacao"], 4), flush=True)
    print("todas as corridas desta etapa feitas")


def bpb(rid):
    p = RAIZ / "runs" / rid / "avaliacao.json"
    return json.loads(p.read_text(encoding="utf-8"))["bpb"] if p.exists() else None


def cmd_analisar(_):
    from scipy.stats import t as student
    plano = json.loads(PLANO.read_text(encoding="utf-8"))
    saida = {}
    for modo in ("autoregressivo", "difusao"):
        for ponto in PONTOS:
            for nome, b in plano["bracos"].items():
                if nome == "B1":
                    continue
                for janela in b["janelas"]:
                    d = [bpb(run_id(nome, janela, ponto, s, modo)) - bpb(run_id("B1", "simbolos", ponto, s, modo))
                         for s in SEMENTES if bpb(run_id(nome, janela, ponto, s, modo)) is not None
                         and bpb(run_id("B1", "simbolos", ponto, s, modo)) is not None]
                    if len(d) < 2:
                        continue
                    d = np.array(d)
                    meia = student.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                    chave = f"{modo}|e{ponto}|{nome}-{janela}"
                    saida[chave] = {"diferencas": d.round(4).tolist(), "media": round(float(d.mean()), 4),
                                    "ic95": [round(float(d.mean() - meia), 4), round(float(d.mean() + meia), 4)]}
                    print(chave, saida[chave])
    (SAIDA / "m10_resultado.json").write_text(json.dumps(saida, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="etapa", required=True)
    sub.add_parser("plano")
    s = sub.add_parser("rodar")
    s.add_argument("--modo", choices=["autoregressivo", "difusao"], default="autoregressivo")
    s.add_argument("--max-min", type=float, default=60)
    sub.add_parser("analisar")
    a = ap.parse_args()
    {"plano": cmd_plano, "rodar": cmd_rodar, "analisar": cmd_analisar}[a.etapa](a)


if __name__ == "__main__":
    main()

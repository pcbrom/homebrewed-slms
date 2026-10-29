#!/usr/bin/env python3
"""Entropy against generative perplexity across sampling temperatures (SPEC section 14, diagnostic).

Generative perplexity alone misleads, since a lower effective temperature lowers it while the
diversity falls (Zheng et al., 2024, section 6.1), so it is reported beside the entropy of each
sample, H = -sum_k p_k log p_k over the token frequencies of the sample (their appendix H.1). The
scorer is external, as GPT-2 Large in MDLM: Tucano 630M at a fixed revision, in fp32. The texts of
the corpus are public domain and may be in its pretraining data, which would lower the perplexity
of copied text; the copy filter of M9 is reported beside it. No decision rests on this figure.

    python scripts/entropia_genppl.py gerar --dispositivo cuda
    <modelos_decisao>/.venv/bin/python scripts/entropia_genppl.py pontuar
    python scripts/entropia_genppl.py resumir
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

PASTA = RAIZ / "resultados" / "entropia_genppl"
TEMPERATURAS = (0.6, 0.8, 1.0, 1.2)
PONTOS = (16, 256)
SEMENTES = (300, 301, 302)
MODOS = ("autoregressivo", "difusao")
N = 64
AVALIADOR = ("TucanoBR/Tucano-630m", "37baa750655d7ee54a542a41a0dc7ffb372135e7")


def run_id(modo, ep, s):
    # descriptive points use seeds 200-204 and the confirmatory point 300-305 in M7
    return f"m7-{modo}-e{ep}-s{s - 100 if ep != 256 else s}"


def entropia(ids: list[int]) -> float:
    c = np.array(list(Counter(ids).values()), dtype=float)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def cmd_gerar(a):
    import torch
    from slm.config import Config
    from slm.corpus import pasta_tokenizador
    from slm.geracao import ate_eos, gerar_ar, gerar_difusao
    from slm.modelo import ConfigRede, Transformer
    from slm.tokenizador import carregar_tokenizador
    PASTA.mkdir(parents=True, exist_ok=True)
    saida = PASTA / "amostras.jsonl"
    feitas = {(r["run_id"], r["temperatura"]) for r in map(json.loads, open(saida, encoding="utf-8"))} if saida.exists() else set()
    with open(saida, "a", encoding="utf-8") as f:
        for ep in PONTOS:
            for modo in MODOS:
                for s in SEMENTES:
                    rid = run_id(modo, ep, s)
                    pasta = RAIZ / "runs" / rid
                    bruto = json.loads((pasta / "config.json").read_text(encoding="utf-8"))
                    cfg = Config.de_dicionario({k: v for k, v in bruto.items() if k in Config.__dataclass_fields__})
                    tok = carregar_tokenizador(pasta_tokenizador(cfg))
                    modelo = Transformer(ConfigRede(vocab_size=cfg.vocab_size, L=cfg.L, n_layer=cfg.n_layer, d=cfg.d, n_head=cfg.n_head))
                    modelo.load_state_dict(torch.load(pasta / "pesos.pt", map_location="cpu"))
                    modelo.to(a.dispositivo).eval()
                    for T in TEMPERATURAS:
                        if (rid, T) in feitas:
                            continue
                        semente = 1000 * s + int(100 * T)
                        ids = (gerar_ar(modelo, cfg.L, n=N, semente=semente, dispositivo=a.dispositivo, temperatura=T) if modo == "autoregressivo"
                               else gerar_difusao(modelo, cfg.L, T_gen=cfg.L, n=N, semente=semente, dispositivo=a.dispositivo, temperatura=T))
                        for j, w in enumerate(ids.tolist()):
                            w = ate_eos([i for i in w if i > 3 or i == 2])
                            f.write(json.dumps({"run_id": rid, "modo": modo, "epocas": ep, "semente": s, "temperatura": T, "j": j,
                                                "tokens": len(w), "entropia": entropia(w) if w else None,
                                                "texto": tok.decode(w).decode("utf-8", "replace")}, ensure_ascii=False) + "\n")
                        f.flush()
                        print(rid, T, flush=True)


def cmd_pontuar(_):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(RAIZ / "scripts"))
    from reguas_publicas import logp_tokens
    repo, rev = AVALIADOR
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    modelo = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=torch.float32).cuda().eval()
    linhas = [json.loads(l) for l in open(PASTA / "amostras.jsonl", encoding="utf-8")]
    seqs = [tok(r["texto"], add_special_tokens=False)["input_ids"] for r in linhas]
    lps = logp_tokens(modelo, tok, seqs)
    with open(PASTA / "pontuadas.jsonl", "w", encoding="utf-8") as f:
        for r, lp in zip(linhas, lps):
            f.write(json.dumps({k: v for k, v in r.items() if k != "texto"} | {"nats_avaliador": float(-lp.sum()), "tokens_avaliador": len(lp)},
                               ensure_ascii=False) + "\n")
    (PASTA / "avaliador.json").write_text(json.dumps({"repositorio": repo, "revisao": rev, "precisao": "fp32",
                                                      "unidade": "token do avaliador, com [EOS]"}, indent=1), encoding="utf-8")


def cmd_resumir(_):
    linhas = [json.loads(l) for l in open(PASTA / "pontuadas.jsonl", encoding="utf-8")]
    res = {}
    for ep in PONTOS:
        for modo in MODOS:
            pts = []
            for T in TEMPERATURAS:
                g = [r for r in linhas if r["epocas"] == ep and r["modo"] == modo and r["temperatura"] == T and r["tokens"] > 0]
                pts.append({"temperatura": T, "amostras": len(g), "entropia": float(np.mean([r["entropia"] for r in g])),
                            "genppl": math.exp(sum(r["nats_avaliador"] for r in g) / sum(r["tokens_avaliador"] for r in g)),
                            "genppl_por_semente": [math.exp(sum(r["nats_avaliador"] for r in g if r["semente"] == s) /
                                                            sum(r["tokens_avaliador"] for r in g if r["semente"] == s)) for s in SEMENTES]})
            res[f"{modo}_e{ep}"] = pts
            print(modo, ep, [(p["temperatura"], round(p["entropia"], 3), round(p["genppl"], 1)) for p in pts])
    (RAIZ / "resultados" / "entropia_genppl.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="etapa", required=True)
    s = sub.add_parser("gerar")
    s.add_argument("--dispositivo", default="cuda")
    sub.add_parser("pontuar")
    sub.add_parser("resumir")
    a = ap.parse_args()
    {"gerar": cmd_gerar, "pontuar": cmd_pontuar, "resumir": cmd_resumir}[a.etapa](a)


if __name__ == "__main__":
    main()

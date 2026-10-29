#!/usr/bin/env python3
"""Longest span copied from the training text, per trained run (memorisation, SPEC level N1).

Loads runs/<run_id>/pesos.pt, draws n samples of L tokens from BOS (autoregressive) or from a
fully masked window (diffusion), cuts each at EOS and measures, with slm.leitura.IndiceTreino, the
longest span of consecutive words found in the training partition. Writes runs/<run_id>/copia.json.

    python scripts/medir_copia.py hp-autoregressivo-lr0.001-do0-e256-s0 --n 16 --dispositivo cpu
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slm.config import Config  # noqa: E402
from slm.corpus import RAIZ, ler_documentos, pasta_tokenizador  # noqa: E402
from slm.geracao import ate_eos, gerar_ar, gerar_difusao  # noqa: E402
from slm.leitura import IndiceTreino  # noqa: E402
from slm.modelo import ConfigRede, Transformer  # noqa: E402
from slm.registro import RegistroCorrida  # noqa: E402
from slm.tokenizador import carregar_tokenizador  # noqa: E402

CAMPOS = set(Config.__dataclass_fields__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_ids", nargs="+")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--semente", type=int, default=1)
    ap.add_argument("--dispositivo", default="cpu")
    a = ap.parse_args()
    indices = {}
    for rid in a.run_ids:
        pasta = RAIZ / "runs" / rid
        bruto = json.loads((pasta / "config.json").read_text(encoding="utf-8"))
        cfg = Config.de_dicionario({k: v for k, v in bruto.items() if k in CAMPOS})
        tok = carregar_tokenizador(pasta_tokenizador(cfg))
        if cfg.corpus not in indices:
            indices[cfg.corpus] = IndiceTreino([d["texto"] for d in ler_documentos(cfg.corpus, "treino")])
        modelo = Transformer(ConfigRede(vocab_size=cfg.vocab_size, L=cfg.L, n_layer=cfg.n_layer, d=cfg.d, n_head=cfg.n_head))
        modelo.load_state_dict(torch.load(pasta / "pesos.pt", map_location="cpu"))
        modelo.to(a.dispositivo).eval()
        if cfg.modo == "autoregressivo":
            amostras = gerar_ar(modelo, cfg.L, n=a.n, semente=a.semente, dispositivo=a.dispositivo)
        else:
            amostras = gerar_difusao(modelo, cfg.L, T_gen=cfg.L, n=a.n, semente=a.semente, dispositivo=a.dispositivo)
        trechos = [indices[cfg.corpus].maior_trecho(tok.decode(ate_eos([i for i in s if i > 3 or i == 2])).decode("utf-8", "replace"))
                   for s in amostras.tolist()]
        res = {"n_amostras": a.n, "tokens_por_amostra": cfg.L, "semente_geracao": a.semente, "maior_trecho_por_amostra": trechos,
               "mediana": float(np.median(trechos)), "maximo": int(max(trechos))}
        RegistroCorrida(pasta).avaliacao(res, "copia.json")
        print(rid, "| mediana", res["mediana"], "| máximo", res["maximo"], "|", trechos, flush=True)


if __name__ == "__main__":
    main()

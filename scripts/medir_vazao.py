#!/usr/bin/env python3
"""Measure training throughput, MFU and peak memory for both modes (gate G1 of the budget).

Runs a few warm-up steps, then times training steps with device synchronisation before each
clock read. MFU = tokens/s x C_train / peak, with C_train from docs/DIMENSIONAMENTO.md.

    python scripts/medir_vazao.py --pico-tflops 104.75 --arq 6x384 --vocab 2048 --dispositivo cuda
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slm.config import Config  # noqa: E402
from slm.modelo import ConfigRede, Transformer  # noqa: E402
from slm.treino import _encaminhar, kernel_atencao, mascara_do_passo, otimizador, perda_lote  # noqa: E402


def custo_por_token(n_layer: int, d: int, L: int, V: int) -> float:
    return 6 * 12 * n_layer * d * d + 6 * n_layer * L * d + 6 * d * V


def medir(modo, arq, V, L, lote, passos, aquecimento, dispositivo, precisao, atencao, pico):
    n_layer, d = (int(x) for x in arq.lower().split("x"))
    cfg = Config(modo=modo, vocab_size=V, L=L, n_layer=n_layer, d=d, n_head=max(1, d // 64), lote=lote,
                 precisao=precisao, atencao=atencao)
    torch.manual_seed(0)
    modelo = Transformer(ConfigRede(vocab_size=V, L=L, n_layer=n_layer, d=d, n_head=cfg.n_head)).to(dispositivo)
    opt = otimizador(modelo, cfg)
    usa_fp16 = precisao == "fp16" and dispositivo.startswith("cuda")
    escalador = torch.amp.GradScaler("cuda", enabled=usa_fp16)
    fwd = _encaminhar(modelo, dispositivo, torch.float16 if usa_fp16 else None)
    w = torch.randint(4, V, (lote, L), device=dispositivo)
    sinc = torch.cuda.synchronize if dispositivo.startswith("cuda") else (lambda: None)
    if dispositivo.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    with kernel_atencao(atencao):
        for i in range(aquecimento + passos):
            if i == aquecimento:
                sinc()
                t0 = time.perf_counter()
            m = mascara_do_passo(cfg, i, lote, L).to(dispositivo) if modo == "difusao" else None
            perda, *_ = perda_lote(fwd, w, modo, m)
            opt.zero_grad(set_to_none=True)
            escalador.scale(perda).backward()
            escalador.step(opt)
            escalador.update()
        sinc()
    seg = time.perf_counter() - t0
    tps = passos * lote * L / seg
    c = custo_por_token(n_layer, d, L, V)
    return {"modo": modo, "arq": arq, "V": V, "L": L, "lote": lote, "precisao": precisao, "atencao": atencao,
            "tokens_por_s": tps, "segundos_por_passo": seg / passos, "flop_por_token": c,
            "mfu": tps * c / (pico * 1e12) if pico else None,
            "memoria_pico_gb": torch.cuda.max_memory_allocated() / 2**30 if dispositivo.startswith("cuda") else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pico-tflops", type=float, required=True)
    ap.add_argument("--arq", default="6x384")
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--passos", type=int, default=50)
    ap.add_argument("--aquecimento", type=int, default=10)
    ap.add_argument("--precisao", default="fp16")
    ap.add_argument("--atencao", default="efficient")
    ap.add_argument("--dispositivo", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    out = [medir(modo, a.arq, a.vocab, a.L, a.lote, a.passos, a.aquecimento, a.dispositivo, a.precisao, a.atencao,
                 a.pico_tflops) for modo in ("autoregressivo", "difusao")]
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

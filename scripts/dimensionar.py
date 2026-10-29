#!/usr/bin/env python3
"""Estimate training throughput, time, data and memory for a small Transformer LM.

Standard library only. All numbers are estimates from peak hardware throughput and an
assumed utilization (MFU); replace the MFU with the value measured on your machine.

Compute per training token follows Kaplan et al. (2020), Table 1, times 3 for
forward plus backward:
    C_train ~= 6 N + 6 n_layer n_ctx d_model + 6 d_model n_vocab
with N = 12 n_layer d_model^2 (non-embedding parameters).

Examples:
    python3 scripts/dimensionar.py --pico-tflops 209.5 --vram-gb 32 --arq 6x384 8x512 12x768
    python3 scripts/dimensionar.py --pico-tflops 65 --vram-gb 16 --arq 4x256 6x384 --horas 1
"""
import argparse
import math

BYTES_PER_PARAM_ADAMW = 16  # fp32 weights, fp32 grads, two fp32 Adam moments


def parse_arch(s):
    layers, d = s.lower().split("x")
    return int(layers), int(d)


def human(x):
    for unit, k in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(x) >= k:
            return f"{x / k:.1f}{unit}"
    return f"{x:.0f}"


def hours(s):
    return f"{s / 60:.0f} min" if s < 3600 else (f"{s / 3600:.1f} h" if s < 86400 * 2 else f"{s / 86400:.1f} d")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pico-tflops", type=float, required=True, help="dense tensor peak in the training precision")
    ap.add_argument("--mfu", type=float, nargs="+", default=[0.3, 0.5], help="assumed or measured utilization")
    ap.add_argument("--vram-gb", type=float, required=True)
    ap.add_argument("--arq", nargs="+", default=["4x256", "6x384", "8x512", "12x768"], help="layers x d_model")
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=8000)
    ap.add_argument("--tokens-por-param", type=float, default=20.0, help="Hoffmann et al. (2022) ~20")
    ap.add_argument("--sem-embedding", action="store_true", help="count only non-embedding params for the data target")
    ap.add_argument("--horas", type=float, help="time budget, prints compute-optimal size for it")
    a = ap.parse_args()

    peak = a.pico_tflops * 1e12
    vram = a.vram_gb * 1024**3
    print(f"pico {a.pico_tflops} TFLOPS | VRAM {a.vram_gb} GB | ctx {a.ctx} | vocab {a.vocab} | "
          f"{a.tokens_por_param:g} tokens/param ({'sem' if a.sem_embedding else 'com'} embedding)\n")
    head = f"{'arq':>7} {'N':>7} {'N+emb':>7} {'FLOP/tok':>9} {'extra':>6} {'mem AdamW':>9} {'tokens alvo':>11}"
    head += "".join(f" {'tok/s@' + str(m):>12} {'tempo@' + str(m):>10}" for m in a.mfu)
    print(head)
    for arch in a.arq:
        nl, d = parse_arch(arch)
        n = 12 * nl * d * d
        n_tot = n + a.vocab * d  # tied input and output embedding
        flop = 6 * n + 6 * nl * a.ctx * d + 6 * d * a.vocab
        extra = flop / (6 * n) - 1
        mem = BYTES_PER_PARAM_ADAMW * n_tot
        target = a.tokens_por_param * (n if a.sem_embedding else n_tot)
        row = f"{arch:>7} {human(n):>7} {human(n_tot):>7} {human(flop):>9} {extra:>6.0%} {mem / vram:>8.1%}V {human(target):>11}"
        for m in a.mfu:
            tps = peak * m / flop
            row += f" {human(tps):>12} {hours(target / tps):>10}"
        print(row)
    print("\nextra: custo de atencao e da camada de saida acima de 6N | mem AdamW: fracao da VRAM antes das ativacoes")

    if a.horas:
        print(f"\norcamento de {a.horas:g} h, aproximacao C = 6 N D com D = {a.tokens_por_param:g} N:")
        for m in a.mfu:
            c = peak * m * a.horas * 3600
            n_opt = math.sqrt(c / (6 * a.tokens_por_param))
            print(f"  MFU {m}: C = {c:.2e} FLOP, N otimo ~ {human(n_opt)}, D ~ {human(a.tokens_por_param * n_opt)} tokens")


if __name__ == "__main__":
    main()

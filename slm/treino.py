"""Training loop, AdamW, fp16 with GradScaler, applied-step budget and sentinel (box 6).

Decisions implemented (docs/decisao_precisao.md, docs/decisao_orcamento.md):
- the forward pass runs under fp16 autocast, the loss is computed in fp32 from logits.float()
  and the backward pass runs outside autocast;
- only applied steps count toward the budget; a step skipped by the GradScaler repeats the same
  batch with the same masks, which keeps the data sequence identical across precision arms;
- the learning rate follows a cosine closed on the run's own length;
- the sentinel compares fp16 and fp32 gradients per tensor on fixed probe batches.
"""
from __future__ import annotations

import contextlib
import math
import time

import torch

from slm.config import Config
from slm.mascaramento import mascara_de_l, sortear_l
from slm.perda import perda_ar, perda_difusao


def kernel_atencao(nome: str):
    if nome == "auto":
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel({"efficient": SDPBackend.EFFICIENT_ATTENTION, "math": SDPBackend.MATH}[nome])


def taxa(passo: int, total: int, cfg: Config) -> float:
    aquec = max(1, int(cfg.aquecimento_frac * total))
    if passo < aquec:
        return cfg.lr * (passo + 1) / aquec
    prog = (passo - aquec) / max(1, total - aquec)
    minimo = cfg.lr * cfg.lr_min_frac
    return minimo + 0.5 * (cfg.lr - minimo) * (1 + math.cos(math.pi * min(1.0, prog)))


def otimizador(modelo: torch.nn.Module, cfg: Config) -> torch.optim.AdamW:
    decai = [p for n, p in modelo.named_parameters() if p.dim() >= 2 and n != "pos"]
    resto = [p for n, p in modelo.named_parameters() if not (p.dim() >= 2 and n != "pos")]
    return torch.optim.AdamW([{"params": decai, "weight_decay": cfg.weight_decay},
                              {"params": resto, "weight_decay": 0.0}], lr=cfg.lr, betas=cfg.betas)


def _encaminhar(modelo, dispositivo: str, dtype):
    def f(ids, causal):
        with torch.autocast(device_type=dispositivo.split(":")[0], dtype=dtype, enabled=dtype is not None):
            return modelo(ids, causal)
    return f


def perda_lote(modelo_f, w: torch.Tensor, modo: str, mascara: torch.Tensor | None, logits: bool = False):
    if modo == "autoregressivo":
        return perda_ar(modelo_f, w, logits)
    return perda_difusao(modelo_f, w, mascara, logits)


def mascara_do_passo(cfg: Config, passo: int, B: int, L: int) -> torch.Tensor:
    """Masks derived from a per-step seed, so a repeated step reuses exactly the same masks."""
    g = torch.Generator().manual_seed(cfg.semente * 1_000_003 + passo)
    return mascara_de_l(sortear_l(B, L, g), L, g)


def _ordem(n: int, total_lotes: int, B: int, g: torch.Generator) -> list[torch.Tensor]:
    """Pre-drawn batch sequence: concatenated permutations, cut into batches of B windows."""
    idx, precisa = [], total_lotes * B
    while sum(len(x) for x in idx) < precisa:
        idx.append(torch.randperm(n, generator=g))
    plano = torch.cat(idx)[:precisa]
    return list(plano.view(total_lotes, B))


def treinar(cfg: Config, modelo: torch.nn.Module, janelas: torch.Tensor, dispositivo: str = "cpu",
            registro=None, sondas: dict | None = None, pontos_sentinela: tuple = (0.0, 0.2, 1.0)) -> dict:
    """Train for cfg.epocas over `janelas` (N, L) counting applied steps only."""
    cfg.validar()
    g = torch.Generator().manual_seed(cfg.semente)
    N, L = janelas.shape
    B = min(cfg.lote, N)
    total = max(1, round(cfg.epocas * N / B))
    plano = _ordem(N, total, B, g)
    usa_fp16 = cfg.precisao == "fp16" and dispositivo.startswith("cuda")
    dtype = torch.float16 if usa_fp16 else None
    escalador = torch.amp.GradScaler("cuda", enabled=usa_fp16)
    opt = otimizador(modelo, cfg)
    fwd = _encaminhar(modelo, dispositivo, dtype)
    marcas = {min(total - 1, int(p * total)) for p in pontos_sentinela}
    modelo.train()
    aplicados, pulados, t0, ultimo = 0, 0, time.time(), float("nan")
    with kernel_atencao(cfg.atencao):
        while aplicados < total:
            w = janelas[plano[aplicados]].to(dispositivo)
            m = mascara_do_passo(cfg, aplicados, B, L).to(dispositivo) if cfg.modo == "difusao" else None
            for grupo in opt.param_groups:
                grupo["lr"] = taxa(aplicados, total, cfg)
            registrar = registro is not None and (aplicados % cfg.registro_cada == 0 or aplicados == total - 1)
            perda, _, *lg = perda_lote(fwd, w, cfg.modo, m, logits=registrar)
            if registrar:
                lg[0].retain_grad()
            opt.zero_grad(set_to_none=True)
            escalador.scale(perda).backward()
            escalador.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), cfg.clip)
            escala = escalador.get_scale() if usa_fp16 else 1.0
            escalador.step(opt)
            escalador.update()
            if usa_fp16 and escalador.get_scale() < escala:
                pulados += 1
                continue
            ultimo = float(perda.detach())
            if registrar:
                gl = lg[0].grad
                registro.passo({"passo": aplicados, "epoca": (aplicados + 1) * B / N, "perda": ultimo,
                                "taxa": opt.param_groups[0]["lr"], "escala_gradscaler": escala,
                                "passos_pulados": pulados,
                                "grad_max_logits": float(gl.abs().max()) if gl is not None else None,
                                "tempo_s": time.time() - t0})
            if sondas and registro is not None and (aplicados in marcas or aplicados % cfg.sentinela_cada == 0):
                registro.sentinela(sentinela(modelo, sondas, cfg.modo, dispositivo, escala, aplicados))
            aplicados += 1
    return {"passos_aplicados": aplicados, "passos_pulados": pulados, "tempo_s": time.time() - t0,
            "perda_final": ultimo, "epocas": aplicados * B / N}


FAIXAS = ((0.0, 0.01), (0.01, 0.1), (0.1, 0.5), (0.5, 1.0))


def sondas_por_faixa(janelas: torch.Tensor, modo: str, n: int = 8, lotes: int = 4, semente: int = 1234) -> dict:
    """Fixed probe batches per band of l / L (one band for the autoregressive mode)."""
    g = torch.Generator().manual_seed(semente)
    L = janelas.shape[1]
    ws = [janelas[torch.randperm(janelas.shape[0], generator=g)[:n]] for _ in range(lotes)]
    if modo == "autoregressivo":
        return {"ar": [(w, None) for w in ws]}
    out = {}
    for a, b in FAIXAS:
        lo = max(1, math.ceil(a * L))
        hi = max(lo, min(L, math.ceil(b * L) - 1 if b < 1 else L))
        out[f"[{a},{b})"] = [(w, mascara_de_l(torch.randint(lo, hi + 1, (w.shape[0],), generator=g), L, g)) for w in ws]
    return out


def _gradientes(modelo, w, m, modo, dispositivo, dtype, escala):
    modelo.zero_grad(set_to_none=True)
    perda, *_ = perda_lote(_encaminhar(modelo, dispositivo, dtype), w, modo, m)
    (perda * escala).backward()
    return {n: (p.grad.detach().float() / escala).clone() for n, p in modelo.named_parameters() if p.grad is not None}


def sentinela(modelo, sondas: dict, modo: str, dispositivo: str, escala: float, passo: int,
              dtype=torch.float16, piso: float = 1e-8) -> list[dict]:
    """Per-tensor relative error and underflow of low-precision gradients against fp32, plus the
    largest activation per block in low precision. Weights are not updated.

    Both passes run in eval mode, so dropout is off and the two gradients come from the same network;
    with dropout on, each pass would draw its own mask and the error would measure the masks, not the
    precision. The previous mode is restored at the end."""
    treinando = modelo.training
    modelo.eval()
    ativ = {}
    ganchos = [b.register_forward_hook(lambda mod, i, o, k=k: ativ.__setitem__(k, max(ativ.get(k, 0.0), float(o.detach().abs().max()))))
               for k, b in enumerate(modelo.blocos)]
    linhas = []
    try:
        for faixa, lotes in sondas.items():
            erros, under, abaixo = {}, {}, {}
            ativ.clear()
            for w, m in lotes:
                w = w.to(dispositivo)
                m = m.to(dispositivo) if m is not None else None
                g_baixa = _gradientes(modelo, w, m, modo, dispositivo, dtype, escala)
                g_32 = _gradientes(modelo, w, m, modo, dispositivo, None, 1.0)
                for nome, g32 in g_32.items():
                    g16, norma = g_baixa[nome], float(g32.norm())
                    if norma > piso:
                        erros.setdefault(nome, []).append(float((g16 - g32).norm()) / norma)
                    else:
                        abaixo[nome] = True
                    under.setdefault(nome, []).append(float(((g16 == 0) & (g32 != 0)).float().mean()))
            for nome in under:
                e = erros.get(nome)
                linhas.append({"passo": passo, "faixa_l": faixa, "tensor": nome,
                               "erro_relativo_p95": float(torch.tensor(e).quantile(0.95)) if e else None,
                               "abaixo_piso": abaixo.get(nome, False),
                               "underflow": float(torch.tensor(under[nome]).quantile(0.95)),
                               "ativacao_max": max(ativ.values()) if ativ else None})
    finally:
        for h in ganchos:
            h.remove()
        modelo.zero_grad(set_to_none=True)
        modelo.train(treinando)
    return linhas

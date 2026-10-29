#!/usr/bin/env python3
"""Where does the Transformer beat the n-gram floor, token by token?

For each autoregressive checkpoint of the budget grid, scores every validation token with the
Transformer and with the n-gram of the order chosen in medir_ngrama.py, on the same windows of L
tokens (context restarts at each window, padding after the first [EOS] left out of both). The
tokens are then split by position in the window, by what the n-gram knows of the context (full
n-gram seen in training, context seen but not the n-gram, context never seen) and by the training
frequency of the token. Finally the two are mixed, p = lam p_T + (1 - lam) p_ng, with lam fitted on
the even documents and measured on the odd ones. Runs on CPU.

    python scripts/diagnostico_ngrama.py --corpus A
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from slm.corpus import ler_documentos, tokenizar  # noqa: E402
from slm.janelas import janelas_documento  # noqa: E402
from slm.modelo import ConfigRede, Transformer  # noqa: E402
from slm.ngrama import BOS, EOS, NGramaKN  # noqa: E402
from slm.perda import entrada_ar  # noqa: E402
from slm.tokenizador import carregar_tokenizador  # noqa: E402

FAIXAS_POS = ((0, 4), (4, 16), (16, 64), (64, 256))
FAIXAS_FREQ = ((0, 10), (10, 100), (100, 1000), (1000, 10**12))


def logp_transformer(pasta: Path, janelas: torch.Tensor) -> np.ndarray:
    cfg = json.loads((pasta / "config.json").read_text(encoding="utf-8"))
    m = Transformer(ConfigRede(vocab_size=cfg["vocab_size"], L=cfg["L"], n_layer=cfg["n_layer"], d=cfg["d"],
                               n_head=cfg["n_head"]))
    m.load_state_dict(torch.load(pasta / "pesos.pt", map_location="cpu"))
    m.eval()
    saida = []
    with torch.no_grad():
        for b in janelas.split(64):
            lp = torch.log_softmax(m(entrada_ar(b), True).double(), -1)
            saida.append(lp.gather(-1, b.unsqueeze(-1)).squeeze(-1).numpy())
    return np.concatenate(saida)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--semente", type=int, default=200)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sufixo = "" if a.corpus == "A" else f"-{a.corpus}"
    ng = json.loads((RAIZ / "resultados" / f"ngrama_{a.corpus}.json").read_text(encoding="utf-8"))
    n = ng["ordem_escolhida"]
    tok = carregar_tokenizador(RAIZ / "tokenizador" / a.corpus)
    treino = [d["ids"] for d in tokenizar(ler_documentos(a.corpus, "treino"), tok)]
    val = tokenizar(ler_documentos(a.corpus, "validacao"), tok)
    modelo_ng = NGramaKN(n, tok.vocab_size).treinar(treino)
    freq = Counter(w for d in treino for w in d)
    vistos_ng, vistos_ctx = modelo_ng.tabelas[n]

    # token table: one row per predicted position up to and including the first [EOS] of the document
    L = 256
    janelas, linhas = [], []
    for k, d in enumerate(val):
        jan = janelas_documento(d["ids"], L)
        seq = list(d["ids"]) + [EOS]
        for j in range(jan.shape[0]):
            janelas.append(jan[j])
            pedaco = seq[j * L:(j + 1) * L]
            for i, w in enumerate(pedaco):
                ctx = tuple(([BOS] * (n - 1) + pedaco[:i])[-(n - 1):])
                estado = "n-grama visto" if ctx + (w,) in vistos_ng else (
                    "contexto visto" if ctx in vistos_ctx else "contexto novo")
                linhas.append((k, len(janelas) - 1, i, w, estado, math.log(modelo_ng.prob(ctx, w))))
    janelas = torch.stack(janelas)
    doc, jan_i, pos, w, estado, lp_ng = (np.array(c) for c in zip(*linhas))
    bytes_tot = sum(d["bytes"] for d in val)
    saida = {"corpus": a.corpus, "ordem_ngrama": n, "tokens": len(w),
             "bpb_ngrama_janelas": float(-lp_ng.sum() / (math.log(2) * bytes_tot)), "orcamentos": {}}
    par = doc % 2 == 0
    for ep in (1, 4, 16, 64, 256):
        pasta = RAIZ / "runs" / f"m7{sufixo}-autoregressivo-e{ep}-s{a.semente}"
        if not (pasta / "pesos.pt").exists():
            continue
        lp_t_jan = logp_transformer(pasta, janelas)
        lp_t = lp_t_jan[jan_i, pos]
        ganho = lp_t - lp_ng  # nats per token gained by the Transformer
        r = {"bpb_transformer_sem_preenchimento": float(-lp_t.sum() / (math.log(2) * bytes_tot)),
             "fracao_tokens_transformer_melhor": float(np.mean(ganho > 0))}

        def quebra(mascaras):
            return {nome: {"fracao_tokens": float(msk.mean()), "ganho_nats_por_token": float(ganho[msk].mean()),
                           "parcela_do_ganho_total_bpb": float(ganho[msk].sum() / (math.log(2) * bytes_tot))}
                    for nome, msk in mascaras.items() if msk.any()}

        r["por_posicao"] = quebra({f"{lo + 1}-{hi}": (pos >= lo) & (pos < hi) for lo, hi in FAIXAS_POS})
        r["por_contexto_ngrama"] = quebra({e: estado == e for e in ("n-grama visto", "contexto visto", "contexto novo")})
        fr = np.array([freq.get(int(x), 0) for x in w])
        r["por_frequencia_treino"] = quebra({f"{lo}-{hi if hi < 10**12 else 'mais'}": (fr >= lo) & (fr < hi)
                                             for lo, hi in FAIXAS_FREQ})
        lams = np.linspace(0, 1, 101)
        mist = lambda lam, m: -np.logaddexp(np.log(lam + 1e-300) + lp_t[m], np.log(1 - lam + 1e-300) + lp_ng[m]).sum()
        lam = float(lams[np.argmin([mist(l, par) for l in lams])])
        b_impar = sum(val[k]["bytes"] for k in range(len(val)) if k % 2 == 1)
        r["mistura"] = {"lambda_ajustado_docs_pares": lam,
                        "bpb_docs_impares": {"transformer": float(-lp_t[~par].sum() / (math.log(2) * b_impar)),
                                             "ngrama": float(-lp_ng[~par].sum() / (math.log(2) * b_impar)),
                                             "mistura": float(mist(lam, ~par) / (math.log(2) * b_impar))}}
        saida["orcamentos"][ep] = r
        print(ep, round(r["bpb_transformer_sem_preenchimento"], 4), "ngrama", round(saida["bpb_ngrama_janelas"], 4),
              "mistura", {k: round(v, 4) for k, v in r["mistura"]["bpb_docs_impares"].items()}, "lam", lam, flush=True)
    (RAIZ / "resultados" / f"diagnostico_ngrama_{a.corpus}.json").write_text(
        json.dumps(saida, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

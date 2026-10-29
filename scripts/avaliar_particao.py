#!/usr/bin/env python3
"""Evaluate trained runs on another partition, with bits per byte without proper names (SPEC 8 and 14).

Writes runs/<run_id>/avaliacao_<particao>.json in the format of avaliacao.json. On corpus A every
document also gets nats_sem_nomes and bytes_sem_nomes: the per-position contributions and the
bytes of the tokens overlapping a name of corpus/A/nomes/<obra>.txt (extracted from the training
partition only) are left out. In the diffusion mode the per-position contribution is the term
(L/l) CE of the number-of-masks NELBO, so the two totals keep the same estimator.

    python scripts/avaliar_particao.py --prefixo m7- --particao teste --dispositivo cuda
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from slm.avaliacao import l_estratificado  # noqa: E402
from slm.config import Config  # noqa: E402
from slm.corpus import ler_documentos, pasta_tokenizador, tokenizar  # noqa: E402
from slm.dados import _RE_PALAVRA  # noqa: E402
from slm.janelas import janelar  # noqa: E402
from slm.mascaramento import mascara_de_l, mascarar  # noqa: E402
from slm.modelo import ConfigRede, Transformer  # noqa: E402
from slm.perda import _ce, entrada_ar  # noqa: E402
from slm.registro import RegistroCorrida  # noqa: E402
from slm.tokenizador import carregar_tokenizador  # noqa: E402

CAMPOS = set(Config.__dataclass_fields__)


@torch.no_grad()
def nats_por_posicao(modelo, janelas, modo, K=1, lote=64, semente=0, dispositivo="cpu"):
    """Per-position nats (N_w, L), drawing masks exactly as slm.avaliacao.nats_por_janela."""
    g = torch.Generator().manual_seed(semente)
    saida = []
    for i in range(0, janelas.shape[0], lote):
        w = janelas[i:i + lote].to(dispositivo)
        if modo == "autoregressivo":
            saida.append(_ce(modelo(entrada_ar(w), True), w).double().cpu())
            continue
        b, L = w.shape
        l = torch.cat([l_estratificado(K, L, g) for _ in range(b)])
        m = mascara_de_l(l, L, g).to(dispositivo)
        wk = w.repeat_interleave(K, dim=0)
        ce = (_ce(modelo(mascarar(wk, m), False), wk) * m).double()
        saida.append((ce * (L / m.sum(dim=1, keepdim=True).double())).cpu().view(b, K, L).mean(dim=1))
    return torch.cat(saida)


def tokens_em_nomes(texto: str, ids: list[int], tok, nomes: frozenset) -> tuple[np.ndarray, np.ndarray]:
    """Bytes of each token and whether it overlaps a name span (byte-level comparison)."""
    tam = np.array([len(tok.decode([i])) for i in ids], dtype=np.int64)
    bruto = texto.encode("utf-8")
    assert tam.sum() == len(bruto), "token bytes do not add up to the document"
    em_nome = np.zeros(len(bruto) + 1, dtype=np.int64)
    for m in _RE_PALAVRA.finditer(texto):
        if m.group(0) in nomes:
            a, b = len(texto[:m.start()].encode("utf-8")), len(texto[:m.end()].encode("utf-8"))
            em_nome[a] += 1
            em_nome[b] -= 1
    cobertos = np.cumsum(em_nome)[:-1] > 0
    fim = np.cumsum(tam)
    acum = np.concatenate([[0], np.cumsum(cobertos)])
    return tam, (acum[fim] - acum[fim - tam]) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-ids", nargs="*", default=[])
    ap.add_argument("--prefixo", default=None, help="every run whose name starts with it")
    ap.add_argument("--particao", choices=["teste", "treino", "validacao"], required=True)
    ap.add_argument("--dispositivo", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    rids = a.run_ids + (sorted(p.name for p in (RAIZ / "runs").iterdir() if p.name.startswith(a.prefixo)) if a.prefixo else [])
    cache = {}
    for rid in rids:
        pasta = RAIZ / "runs" / rid
        arq = pasta / f"avaliacao_{a.particao}.json"
        if arq.exists() or not (pasta / "pesos.pt").exists():
            continue
        cfg = Config.de_dicionario({k: v for k, v in json.loads((pasta / "config.json").read_text(encoding="utf-8")).items() if k in CAMPOS})
        chave = (cfg.corpus, str(pasta_tokenizador(cfg)))
        if chave not in cache:
            tok = carregar_tokenizador(chave[1])
            docs = tokenizar(ler_documentos(cfg.corpus, a.particao), tok)
            marcas = None
            if cfg.corpus == "A" and a.particao != "treino":
                lista = {p.stem: frozenset(p.read_text(encoding="utf-8").split("\n")) - {""} for p in (RAIZ / "corpus" / "A" / "nomes").glob("*.txt")}
                marcas = [tokens_em_nomes(d["texto"], d["ids"], tok, lista[d["grupo"]]) for d in docs]
            cache[chave] = (docs, marcas)
        docs, marcas = cache[chave]
        modelo = Transformer(ConfigRede(vocab_size=cfg.vocab_size, L=cfg.L, n_layer=cfg.n_layer, d=cfg.d, n_head=cfg.n_head))
        modelo.load_state_dict(torch.load(pasta / "pesos.pt", map_location="cpu"))
        modelo.to(a.dispositivo).eval()
        janelas, doc = janelar([d["ids"] for d in docs], cfg.L)
        pos = nats_por_posicao(modelo, janelas, cfg.modo, dispositivo=a.dispositivo)
        linhas, inicio = [], 0
        for j, d in enumerate(docs):
            nw = int((doc == j).sum())
            p = pos[inicio:inicio + nw].reshape(-1)
            inicio += nw
            r = {"doc_id": d["doc_id"], "grupo": d["grupo"], "nats": float(p.sum()), "bytes": int(d["bytes"]), "janelas": nw}
            if marcas:
                tam, em = marcas[j]
                r["nats_sem_nomes"] = r["nats"] - float(p[: len(d["ids"])][torch.from_numpy(em)].sum())
                r["bytes_sem_nomes"] = int(d["bytes"] - tam[em].sum())
            linhas.append(r)
        res = {"conjunto": a.particao, "modo": cfg.modo, "K": 1, "documentos": linhas,
               "bpb": sum(r["nats"] for r in linhas) / (np.log(2) * sum(r["bytes"] for r in linhas))}
        if marcas:
            res["bpb_sem_nomes"] = sum(r["nats_sem_nomes"] for r in linhas) / (np.log(2) * sum(r["bytes_sem_nomes"] for r in linhas))
            res["fracao_bytes_em_nomes"] = 1 - sum(r["bytes_sem_nomes"] for r in linhas) / sum(r["bytes"] for r in linhas)
        RegistroCorrida(pasta).avaliacao(res, arq.name)
        print(rid, a.particao, round(res["bpb"], 4), round(res.get("bpb_sem_nomes", float("nan")), 4), flush=True)


if __name__ == "__main__":
    main()

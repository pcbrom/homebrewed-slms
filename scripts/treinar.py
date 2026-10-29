#!/usr/bin/env python3
"""Train one run and evaluate it on the validation partition.

Writes runs/<run_id>/ as in docs/CONTRATO_DADO.md: config.json, passos.jsonl, sentinela.jsonl
(with --sentinela), avaliacao.json and pesos.pt (kept out of git). A run directory is never
overwritten.

    python scripts/treinar.py --modo difusao --epocas 16 --semente 0 --papel calibracao --dispositivo cuda
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slm.avaliacao import avaliar, escolher_K  # noqa: E402
from slm.config import Config  # noqa: E402
from slm.corpus import RAIZ, fracao_grupos, hash_corpus, ler_documentos, tokenizar  # noqa: E402
from slm.janelas import janelar  # noqa: E402
from slm.modelo import ConfigRede, Transformer  # noqa: E402
from slm.registro import RegistroCorrida  # noqa: E402
from slm.tokenizador import carregar_tokenizador, hash_tokenizador  # noqa: E402
from slm.treino import sondas_por_faixa, treinar  # noqa: E402


def construir_modelo(cfg: Config) -> Transformer:
    torch.manual_seed(cfg.semente)
    return Transformer(ConfigRede(vocab_size=cfg.vocab_size, L=cfg.L, n_layer=cfg.n_layer, d=cfg.d,
                                  n_head=cfg.n_head, dropout=cfg.dropout))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="JSON with Config fields; command-line values override it")
    ap.add_argument("--modo", choices=["difusao", "autoregressivo"])
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--epocas", type=float)
    ap.add_argument("--semente", type=int)
    ap.add_argument("--fracao-treino", type=float, help="train on this fraction of the bytes, whole groups, nested")
    ap.add_argument("--papel")
    ap.add_argument("--precisao", choices=["fp16", "fp32"])
    ap.add_argument("--atencao", choices=["efficient", "math", "auto"])
    ap.add_argument("--lr", type=float)
    ap.add_argument("--lote", type=int)
    ap.add_argument("--dropout", type=float)
    ap.add_argument("--run-id")
    ap.add_argument("--dispositivo", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--orcamento-avaliacao", type=int, default=0, help="C in network passes; 0 means K=1")
    ap.add_argument("--sentinela", action="store_true")
    ap.add_argument("--tokenizador", default=None, help="default: tokenizador/<corpus>")
    a = ap.parse_args()

    # fp32 reference means true fp32: no TF32 in matmuls or convolutions
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = json.loads(Path(a.config).read_text(encoding="utf-8")) if a.config else {}
    corpus = a.corpus or base.get("corpus", "A")
    pasta_tok = Path(a.tokenizador) if a.tokenizador else RAIZ / "tokenizador" / corpus
    tok = carregar_tokenizador(pasta_tok)
    for campo in ("modo", "corpus", "epocas", "semente", "papel", "precisao", "atencao", "lr", "lote", "dropout"):
        if getattr(a, campo) is not None:
            base[campo] = getattr(a, campo)
    base.setdefault("corpus", "A")
    base["vocab_size"] = tok.vocab_size
    base["corpus_hash"] = hash_corpus(base["corpus"])
    base["tokenizador_hash"] = hash_tokenizador(tok)
    if a.fracao_treino:
        base.setdefault("extra", {})["fracao_treino"] = a.fracao_treino
    if a.tokenizador:
        base.setdefault("extra", {})["tokenizador"] = str(pasta_tok.resolve().relative_to(RAIZ)) if pasta_tok.resolve().is_relative_to(RAIZ) else str(pasta_tok)
    cfg = Config.de_dicionario(base)
    cfg.run_id = a.run_id or f"{cfg.papel}-{cfg.modo}-e{cfg.epocas:g}-s{cfg.semente}"

    pasta = RAIZ / "runs" / cfg.run_id
    if pasta.exists():
        sys.exit(f"{pasta} exists: runs are never overwritten")
    reg = RegistroCorrida(pasta)
    reg.config(cfg.dicionario())

    docs_treino = ler_documentos(cfg.corpus, "treino")
    if cfg.extra.get("fracao_treino"):
        docs_treino = fracao_grupos(docs_treino, cfg.extra["fracao_treino"])
    treino = tokenizar(docs_treino, tok)
    validacao = tokenizar(ler_documentos(cfg.corpus, "validacao"), tok)
    janelas, _ = janelar([d["ids"] for d in treino], cfg.L)
    modelo = construir_modelo(cfg).to(a.dispositivo)
    sondas = sondas_por_faixa(janelas, cfg.modo) if a.sentinela else None
    resumo = treinar(cfg, modelo, janelas, a.dispositivo, reg, sondas)
    torch.save(modelo.state_dict(), pasta / "pesos.pt")

    modelo.eval()
    n_val = sum(janelar([d["ids"]], cfg.L)[0].shape[0] for d in validacao)
    K = escolher_K(a.orcamento_avaliacao, n_val) if a.orcamento_avaliacao else 1
    res = avaliar(modelo, validacao, cfg.modo, cfg.L, K=K, dispositivo=a.dispositivo)
    res["resumo_treino"] = resumo
    reg.avaliacao(res)
    print(json.dumps({"run_id": cfg.run_id, "bpb_validacao": res["bpb"], **resumo}, indent=2))


if __name__ == "__main__":
    main()

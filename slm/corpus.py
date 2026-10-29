"""Read a built corpus and tokenize its partitions (CONTRATO_DADO section 1)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def ler_documentos(corpus: str, particao: str | None = None, raiz: Path = RAIZ) -> list[dict]:
    caminho = raiz / "corpus" / corpus / "documentos.jsonl"
    with open(caminho, encoding="utf-8") as f:
        docs = [json.loads(l) for l in f if l.strip()]
    return [d for d in docs if particao is None or d["particao"] == particao]


def hash_corpus(corpus: str, raiz: Path = RAIZ) -> str:
    return hashlib.sha256((raiz / "corpus" / corpus / "documentos.jsonl").read_bytes()).hexdigest()


def tokenizar(docs: list[dict], tok) -> list[dict]:
    """Add the token ids to each document, keeping the byte count of the NFC text."""
    return [dict(d, ids=tok.encode(d["texto"])) for d in docs]


def pasta_tokenizador(cfg) -> Path:
    """Tokenizer folder of a run: the one recorded in cfg.extra, else tokenizador/<corpus>."""
    p = (getattr(cfg, "extra", None) or {}).get("tokenizador")
    return (RAIZ / p if not Path(p).is_absolute() else Path(p)) if p else RAIZ / "tokenizador" / cfg.corpus



def fracao_grupos(docs: list[dict], fracao: float, semente: int = 0) -> list[dict]:
    """Documents of the smallest prefix of groups, in a fixed hash order, holding at least the
    fraction of the bytes. The order depends only on the group and the seed, so the subset of a
    smaller fraction is contained in that of a larger one (nested data curve)."""
    if not 0 < fracao <= 1:
        raise ValueError("fracao deve estar em (0, 1]")
    por_grupo: dict[str, list[dict]] = {}
    for d in docs:
        por_grupo.setdefault(d["grupo"], []).append(d)
    ordem = sorted(por_grupo, key=lambda g: hashlib.sha256(f"{semente}:{g}".encode()).hexdigest())
    alvo, soma, saida = fracao * sum(d["bytes"] for d in docs), 0, []
    for g in ordem:
        if soma >= alvo:
            break
        saida += por_grupo[g]
        soma += sum(d["bytes"] for d in por_grupo[g])
    return saida

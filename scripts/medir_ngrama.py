#!/usr/bin/env python3
"""Floor reference: n-gram with interpolated modified Kneser-Ney, in bits per byte.

The order is chosen on the validation partition among 3 to 6 (decisao_avaliacao.md), then the
chosen model is measured on validation, test and, when given, an external corpus tokenized with
the same tokenizer. Every document is scored with its [EOS], as the Transformer predicts it.

    python scripts/medir_ngrama.py --corpus A
    python scripts/medir_ngrama.py --corpus B --externo B_externo
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from slm.corpus import ler_documentos, tokenizar  # noqa: E402
from slm.ngrama import NGramaKN  # noqa: E402
from slm.tokenizador import carregar_tokenizador  # noqa: E402

ORDENS = (3, 4, 5, 6)


def bpb(modelo, docs):
    return sum(modelo.nats(d["ids"]) for d in docs) / (math.log(2) * sum(d["bytes"] for d in docs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--externo", help="corpus scored with the same tokenizer, all partitions")
    a = ap.parse_args()
    tok = carregar_tokenizador(RAIZ / "tokenizador" / a.corpus)
    part = {p: tokenizar(ler_documentos(a.corpus, p), tok) for p in ("treino", "validacao", "teste")}
    treino = [d["ids"] for d in part["treino"]]
    saida = {"corpus": a.corpus, "vocab_size": tok.vocab_size, "ordens": {}}
    modelos = {}
    for n in ORDENS:
        t0 = time.time()
        modelos[n] = NGramaKN(n, tok.vocab_size).treinar(treino)
        saida["ordens"][n] = {"bpb_validacao": bpb(modelos[n], part["validacao"]), "segundos": time.time() - t0}
        print(n, round(saida["ordens"][n]["bpb_validacao"], 4), round(saida["ordens"][n]["segundos"], 1), flush=True)
    n = min(ORDENS, key=lambda k: saida["ordens"][k]["bpb_validacao"])
    saida["ordem_escolhida"] = n
    saida["bpb_teste"] = bpb(modelos[n], part["teste"])
    if a.externo:
        saida["externo"] = {"corpus": a.externo, "bpb": bpb(modelos[n], tokenizar(ler_documentos(a.externo), tok))}
    print(json.dumps({k: v for k, v in saida.items() if k != "ordens"}, ensure_ascii=False))
    (RAIZ / "resultados").mkdir(exist_ok=True)
    (RAIZ / "resultados" / f"ngrama_{a.corpus}.json").write_text(json.dumps(saida, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

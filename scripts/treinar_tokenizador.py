#!/usr/bin/env python3
"""Train the reference tokenizer on the training partition and pick V on the validation text.

Decision: docs/decisao_tokenizador.md. V is the largest value of the grid with at most
`--limiar` of learned tokens absent from the validation text. The didactic BPE is checked
against the library on a sample (merges and ids must be identical).

    python scripts/treinar_tokenizador.py --corpus A
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slm.corpus import RAIZ, ler_documentos  # noqa: E402
from slm.tokenizador import BPEBiblioteca, BPEDidatico, escolher_vocabulario, fracao_ausente  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="A")
    ap.add_argument("--grade", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    ap.add_argument("--limiar", type=float, default=0.10)
    ap.add_argument("--amostra-bytes", type=int, default=440_000, help="size of the didactic sample")
    ap.add_argument("--saida", default=None, help="default: tokenizador/<corpus>")
    a = ap.parse_args()

    treino = [d["texto"] for d in ler_documentos(a.corpus, "treino")]
    retido = [d["texto"] for d in ler_documentos(a.corpus, "validacao")]
    V, medidas = escolher_vocabulario(treino, retido, tuple(a.grade), a.limiar)
    tok = BPEBiblioteca().treinar(treino, V)
    tok._meta.update(fracao_ausente_retido=fracao_ausente(tok, retido), bytes_retido=sum(len(t.encode()) for t in retido),
                     medidas_regra_V={str(k): v for k, v in medidas.items()}, limiar_regra_V=a.limiar)

    amostra, n = [], 0
    for t in treino:
        if n >= a.amostra_bytes:
            break
        amostra.append(t)
        n += len(t.encode())
    t0 = time.time()
    did = BPEDidatico().treinar(amostra, V)
    seg = time.time() - t0
    lib = BPEBiblioteca().treinar(amostra, V)
    iguais_fusoes = did.merges == lib.merges()
    iguais_ids = all(did.encode(t) == lib.encode(t) for t in retido)
    tok._meta.update(didatico={"bytes_amostra": n, "segundos_treino": round(seg, 1),
                               "fusoes_iguais": iguais_fusoes, "ids_iguais_no_retido": iguais_ids})
    tok.salvar(Path(a.saida) if a.saida else RAIZ / "tokenizador" / a.corpus)
    print(json.dumps({"V": V, "medidas": medidas, **tok._meta["didatico"]}, indent=2))
    if not (iguais_fusoes and iguais_ids):
        sys.exit("didactic BPE and library diverge: gate G4 of the tokenizer fails")


if __name__ == "__main__":
    main()

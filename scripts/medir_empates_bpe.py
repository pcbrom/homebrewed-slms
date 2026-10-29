#!/usr/bin/env python3
"""Count BPE merges decided by the tie-break convention (docs/decisao_tokenizador.md).

Repeats the training loop of BPEDidatico on the same sample used by scripts/treinar_tokenizador.py
and counts, at every merge, how many pairs share the maximum count. A merge with more than one
such pair is decided only by the tie-break convention, which is why two implementations without a
written convention can diverge. Writes tokenizador/<corpus>/empates_bpe.json.

    python scripts/medir_empates_bpe.py --corpus A
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import regex

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slm.corpus import RAIZ, ler_documentos  # noqa: E402
from slm.tokenizador import GPT2_REGEX, N_ESP, ORDEM_BYTES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="A")
    ap.add_argument("--amostra-bytes", type=int, default=440_000)
    a = ap.parse_args()
    meta = json.loads((RAIZ / "tokenizador" / a.corpus / "metadados.json").read_text(encoding="utf-8"))
    V = meta["vocab_size"]
    amostra, n = [], 0
    for d in ler_documentos(a.corpus, "treino"):
        if n >= a.amostra_bytes:
            break
        amostra.append(d["texto"])
        n += len(d["texto"].encode())
    pat = regex.compile(GPT2_REGEX)
    palavras = collections.Counter(tuple(bytes([b]) for b in w.encode("utf-8")) for t in amostra for w in pat.findall(t))
    vocab = {bytes([b]): N_ESP + i for i, b in enumerate(ORDEM_BYTES)}
    empatadas, maior_empate, primeira = 0, 0, None
    while N_ESP + len(vocab) < V:
        stats = collections.Counter()
        for w, c in palavras.items():
            for p in zip(w, w[1:]):
                stats[p] += c
        if not stats:
            break
        topo = max(stats.values())
        k = sum(1 for v in stats.values() if v == topo)
        if k > 1:
            empatadas += 1
            maior_empate = max(maior_empate, k)
            primeira = primeira or len(vocab) + N_ESP - 260 + 1
        best = max(stats, key=lambda p: (stats[p], -vocab[p[0]], -vocab[p[1]]))
        vocab[best[0] + best[1]] = len(vocab) + N_ESP
        novo = collections.Counter()
        for w, c in palavras.items():
            out, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                    out.append(w[i] + w[i + 1])
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            novo[tuple(out)] += c
        palavras = novo
    fusoes = N_ESP + len(vocab) - 260
    res = {"corpus": a.corpus, "V": V, "bytes_amostra": n, "fusoes": fusoes, "fusoes_com_empate_no_topo": empatadas,
           "fracao": round(empatadas / fusoes, 4), "primeira_fusao_com_empate": primeira, "maior_empate": maior_empate}
    (RAIZ / "tokenizador" / a.corpus / "empates_bpe.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

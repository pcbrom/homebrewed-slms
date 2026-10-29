#!/usr/bin/env python3
"""Density of Morse-style variable-length codes against the reference BPE.

For each unit (character, word, morpheme proxy, mixed) and code alphabet size A, codes are built on
the training partition and measured on the validation partition:
  rank+sep  hand-designable rule: units sorted by frequency, the first A get 1 symbol, the next A^2
            get 2, and so on; every unit is followed by a separator symbol (alphabet A + 1)
  huffman   A-ary Huffman code lengths, prefix-free and without separator; the bound any hand-made
            prefix-free table with the same unit and alphabet can reach
Units unseen in training pay an escape unit and are spelled out character by character.

Reported per variant: symbols per byte, nominal bits per byte (symbols x log2 of the alphabet size
per byte), unigram cross-entropy of the units per byte (train frequencies, add-one on the escape
path) and information per bit (cross-entropy over nominal bits).

    .venv/bin/python experimentos/m10_codigo/medidas_codigo.py
"""
import collections
import heapq
import json
import math
import sys
from pathlib import Path

import regex

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
from slm.corpus import ler_documentos  # noqa: E402
from slm.tokenizador import GPT2_REGEX, BPEBiblioteca  # noqa: E402

PAT = regex.compile(GPT2_REGEX)
# Morpheme proxy: common Portuguese suffixes, longest first, in the period spelling too.
SUFIXOS = sorted({
    "mente", "ção", "ções", "cção", "cções", "são", "sões", "dade", "dades", "ismo", "ismos", "ista", "istas",
    "ável", "áveis", "ível", "íveis", "oso", "osa", "osos", "osas", "inho", "inha", "inhos", "inhas",
    "ando", "endo", "indo", "ado", "ada", "ados", "adas", "ido", "ida", "idos", "idas",
    "ava", "avas", "ávamos", "avam", "ia", "ias", "íamos", "iam", "ou", "eu", "iu", "aram", "eram", "iram",
    "ar", "er", "ir", "ará", "erá", "irá", "aria", "eria", "iria", "asse", "esse", "isse",
    "eiro", "eira", "eiros", "eiras", "ez", "eza", "ura", "uras", "agem", "mento", "mentos",
    "es", "s", "a", "o", "e",
}, key=len, reverse=True)
ESC = "\x00ESC"


def morfemas(piece: str) -> list[str]:
    """Strip up to two suffixes from an alphabetic piece, keeping a stem of at least 3 letters."""
    lead = " " if piece.startswith(" ") else ""
    w = piece[len(lead):]
    if not w.isalpha() or len(w) < 5:
        return [piece]
    out = []
    for _ in range(2):
        for s in SUFIXOS:
            if w.lower().endswith(s) and len(w) - len(s) >= 3:
                out.insert(0, w[-len(s):])
                w = w[:-len(s)]
                break
        else:
            break
    return [lead + w] + out


def segmentar(textos, unidade, topo=None):
    for t in textos:
        pieces = PAT.findall(t)
        assert "".join(pieces) == t
        for p in pieces:
            if unidade == "caractere":
                yield from p
            elif unidade == "palavra":
                yield p
            elif unidade == "morfema":
                yield from morfemas(p)
            elif unidade == "mista":
                if p in topo:
                    yield p
                else:
                    yield from morfemas(p)


def comprimentos_rank(n, A):
    out, l, cap = [], 1, A
    for r in range(n):
        while r >= cap:
            l += 1
            cap += A ** l
        out.append(l)
    return out


def comprimentos_huffman(freqs, A):
    """A-ary Huffman code lengths (dummy zero-weight leaves pad to (n - 1) % (A - 1) == 0)."""
    n = len(freqs)
    pad = (A - 1 - (n - 1) % (A - 1)) % (A - 1) if A > 2 else 0
    heap = [(f, i, None) for i, f in enumerate(freqs)] + [(0, n + j, None) for j in range(pad)]
    heapq.heapify(heap)
    filhos, cont = {}, n + pad
    while len(heap) > 1:
        grupo = [heapq.heappop(heap) for _ in range(min(A, len(heap)))]
        filhos[cont] = [g[1] for g in grupo]
        heapq.heappush(heap, (sum(g[0] for g in grupo), cont, None))
        cont += 1
    prof = {heap[0][1]: 0}
    pilha = [heap[0][1]]
    while pilha:
        no = pilha.pop()
        for c in filhos.get(no, []):
            prof[c] = prof[no] + 1
            pilha.append(c)
    return [max(1, prof[i]) for i in range(n)]


def medir(unidade, treino, valid, bytes_valid, alfabetos, topo=None):
    cont = collections.Counter(segmentar(treino, unidade, topo))
    chars = collections.Counter(c for t in treino for c in t)
    for c in chars:
        cont.setdefault(c, 0)
    cont[ESC] = 1
    inventario = [u for u, _ in cont.most_common()]
    idx = {u: i for i, u in enumerate(inventario)}
    total = sum(cont.values()) + len(cont)
    logp = {u: math.log2((cont[u] + 1) / total) for u in inventario}
    seq, oov = [], 0
    for u in segmentar(valid, unidade, topo):
        if u in idx:
            seq.append(u)
        else:
            oov += 1
            seq.append(ESC)
            seq.extend(c if c in idx else ESC for c in u)
    xent = -sum(logp[u] for u in seq)
    contagem = collections.Counter(seq)
    linhas = []
    for A in alfabetos:
        for esquema in ("rank+sep", "huffman"):
            if esquema == "rank+sep":
                L = comprimentos_rank(len(inventario), A)
                simb = sum(contagem[u] * (L[idx[u]] + 1) for u in contagem)
                alfa = A + 1
            else:
                if len(inventario) > 200_000:
                    continue
                L = comprimentos_huffman([cont[u] + 1 for u in inventario], A)
                simb = sum(contagem[u] * L[idx[u]] for u in contagem)
                alfa = A
            nominal = simb * math.log2(alfa)
            linhas.append({"unidade": unidade, "A": A, "esquema": esquema, "inventario": len(inventario),
                           "oov_valid": round(oov / max(1, len(seq)), 4), "simbolos_por_byte": round(simb / bytes_valid, 4),
                           "bits_nominais_por_byte": round(nominal / bytes_valid, 3),
                           "xent_unidades_por_byte": round(xent / bytes_valid, 3), "info_por_bit": round(xent / nominal, 3)})
    return linhas


def main():
    treino = [d["texto"] for d in ler_documentos("A", "treino")]
    valid = [d["texto"] for d in ler_documentos("A", "validacao")]
    B = sum(len(t.encode("utf-8")) for t in valid)
    alfabetos = (2, 16, 256, 4096)
    res = []
    palavras = collections.Counter(p for t in treino for p in PAT.findall(t))
    for unidade in ("caractere", "palavra", "morfema"):
        res += medir(unidade, treino, valid, B, alfabetos)
        print(unidade, "ok", flush=True)
    for n in (256, 1024, 4096):
        linhas = medir("mista", treino, valid, B, alfabetos, {p for p, _ in palavras.most_common(n)})
        for r in linhas:
            r["unidade"] = f"mista {n}"
        res += linhas
        print("mista", n, "ok", flush=True)
    for V in (1024, 4096, 8192):
        tok = BPEBiblioteca().treinar(treino, V)
        ids = [i for t in valid for i in tok.encode(t)]
        c = collections.Counter(i for t in treino for i in tok.encode(t))
        tot = sum(c.values()) + V
        xent = -sum(math.log2((c[i] + 1) / tot) for i in ids)
        nominal = len(ids) * math.log2(V)
        res.append({"unidade": f"BPE V={V}", "A": V, "esquema": "BPE", "inventario": V, "oov_valid": 0.0,
                    "simbolos_por_byte": round(len(ids) / B, 4), "bits_nominais_por_byte": round(nominal / B, 3),
                    "xent_unidades_por_byte": round(xent / B, 3), "info_por_bit": round(xent / nominal, 3)})
    (RAIZ / "resultados" / "m10" / "medidas_codigo.json").write_text(json.dumps({"bytes_validacao": B, "linhas": res}, indent=1, ensure_ascii=False), encoding="utf-8")
    cab = ["unidade", "A", "esquema", "inventario", "oov_valid", "simbolos_por_byte", "bits_nominais_por_byte", "xent_unidades_por_byte", "info_por_bit"]
    print(" | ".join(cab))
    for r in res:
        print(" | ".join(str(r[k]) for k in cab))


if __name__ == "__main__":
    main()

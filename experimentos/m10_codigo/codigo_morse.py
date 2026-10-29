"""Morse-style variable-length code over mixed units, honouring the tokenizer contract.

Units: the `topo` most frequent pre-tokens of the training text are kept whole, the others are split
by the hand-designed morpheme rule (experimentos/m10_codigo/morfema.py). Every unit gets an A-ary prefix-free
codeword from an A-ary Huffman code built on training frequencies and frozen. A unit absent from the
table is spelled between two ESC codewords, character by character with the training characters and as
UTF-8 bytes for characters never seen, so encode/decode is exact for any text.
Ids: 0..3 are the reserved specials, symbol k of the code alphabet is id 4 + k.

    python experimentos/m10_codigo/codigo_morse.py construir --A 256 --topo 4096 --saida tokenizador/m10_C1
    python experimentos/m10_codigo/codigo_morse.py verificar tokenizador/m10_C1
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import heapq
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import regex

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(AQUI))
import morfema  # noqa: E402
from slm.tokenizador import ESPECIAIS, GPT2_REGEX, N_ESP  # noqa: E402

PAT = regex.compile(GPT2_REGEX)
ESC = "\x00ESC"
BYTE = "\x00B{:02x}"
CHAR = "\x00C"
SUBSTITUICAO = "�".encode("utf-8")


def unidades(texto: str, topo: set[str], variante: str) -> list[str]:
    """Mixed units; variante "aproximacao" uses the crude suffix-list proxy of medidas_codigo."""
    if variante == "aproximacao":
        import medidas_codigo
        seg = medidas_codigo.morfemas
    else:
        seg = lambda p: morfema.segmentar(p, variante)  # noqa: E731
    out = []
    for p in PAT.findall(texto):
        out += [p] if p in topo else seg(p)
    return out


def comprimentos_huffman(pesos: list[int], A: int) -> list[int]:
    n = len(pesos)
    pad = (A - 1 - (n - 1) % (A - 1)) % (A - 1) if A > 2 else 0
    heap = [(w, i) for i, w in enumerate(pesos)] + [(0, n + j) for j in range(pad)]
    heapq.heapify(heap)
    filhos, prox = {}, n + pad
    while len(heap) > 1:
        grupo = [heapq.heappop(heap) for _ in range(min(A, len(heap)))]
        filhos[prox] = [g[1] for g in grupo]
        heapq.heappush(heap, (sum(g[0] for g in grupo), prox))
        prox += 1
    prof, pilha = {heap[0][1]: 0}, [heap[0][1]]
    while pilha:
        no = pilha.pop()
        for c in filhos.get(no, ()):
            prof[c] = prof[no] + 1
            pilha.append(c)
    return [max(1, prof[i]) for i in range(n)]


def canonico(comprimentos: list[int], A: int) -> list[tuple[int, ...]]:
    """Canonical A-ary prefix-free codewords for the given lengths (Kraft sum <= 1)."""
    ordem = sorted(range(len(comprimentos)), key=lambda i: (comprimentos[i], i))
    codigos, valor, atual = [None] * len(comprimentos), 0, comprimentos[ordem[0]]
    for i in ordem:
        valor *= A ** (comprimentos[i] - atual)
        atual = comprimentos[i]
        digs, v = [], valor
        for _ in range(atual):
            digs.append(v % A)
            v //= A
        assert v == 0, "Kraft inequality violated"
        codigos[i] = tuple(reversed(digs))
        valor += 1
    return codigos


class CodigoMorse:
    especiais = dict(ESPECIAIS)

    def __init__(self, inventario: list[str], codigos: list[tuple[int, ...]], A: int, topo: list[str], variante: str, meta: dict):
        self.A, self.variante, self.topo = A, variante, set(topo)
        self._topo_lista = topo
        self.inventario, self.codigos = inventario, codigos
        self.cod = dict(zip(inventario, codigos))
        self.trie = {c: u for u, c in zip(inventario, codigos)}
        self.max_len = max(len(c) for c in codigos)
        self._meta = meta
        self.invalidos = 0

    @property
    def vocab_size(self) -> int:
        return N_ESP + self.A

    @classmethod
    def construir(cls, treino: Iterable[str], A: int, n_topo: int, variante: str = "prefixo") -> "CodigoMorse":
        treino = list(treino)
        palavras = collections.Counter(p for t in treino for p in PAT.findall(t))
        topo = [p for p, _ in palavras.most_common(n_topo)]
        cont = collections.Counter(u for t in treino for u in unidades(t, set(topo), variante))
        cont[ESC] += 1
        for c, k in collections.Counter(c for t in treino for c in t).items():
            cont[CHAR + c] += k
        for b in range(256):
            cont[BYTE.format(b)] += 1
        inventario = [u for u, _ in sorted(cont.items(), key=lambda kv: (-kv[1], kv[0]))]
        codigos = canonico(comprimentos_huffman([cont[u] for u in inventario], A), A)
        meta = {"A": A, "n_topo": n_topo, "variante": variante, "inventario": len(inventario)}
        return cls(inventario, codigos, A, topo, variante, meta)

    def encode(self, texto: str) -> list[int]:
        out = []
        for u in unidades(texto, self.topo, self.variante):
            c = self.cod.get(u)
            if c is None:
                out += self.cod[ESC]
                for ch in u:
                    cc = self.cod.get(CHAR + ch)
                    if cc is None:
                        for b in ch.encode("utf-8"):
                            out += self.cod[BYTE.format(b)]
                    else:
                        out += cc
                out += self.cod[ESC]
            else:
                out += c
        return [N_ESP + d for d in out]

    def decode(self, ids: Iterable[int]) -> bytes:
        out, buf, escape = bytearray(), (), False
        for i in ids:
            if i < N_ESP:
                continue
            buf += (i - N_ESP,)
            u = self.trie.get(buf)
            if u is None:
                if len(buf) >= self.max_len:
                    self.invalidos += 1
                    out += SUBSTITUICAO
                    buf = ()
                continue
            buf = ()
            if u == ESC:
                escape = not escape
            elif u.startswith("\x00B"):
                out.append(int(u[2:], 16))
            elif u.startswith(CHAR):
                out += u[len(CHAR):].encode("utf-8")
            else:
                out += u.encode("utf-8")
        return bytes(out)

    def salvar(self, pasta: Path) -> None:
        pasta = Path(pasta)
        pasta.mkdir(parents=True, exist_ok=True)
        tabela = {"A": self.A, "variante": self.variante, "topo": self._topo_lista,
                  "inventario": self.inventario, "codigos": ["".join(f"{d:04x}" for d in c) for c in self.codigos]}
        (pasta / "tabela.json").write_text(json.dumps(tabela, ensure_ascii=False), encoding="utf-8")
        meta = {**self.metadados(), "modulo": os.path.relpath(Path(__file__).resolve(), pasta.resolve()),
                "classe": "CodigoMorse", "sha256": hashlib.sha256((pasta / "tabela.json").read_bytes()).hexdigest()}
        (pasta / "metadados.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def carregar(cls, pasta: Path) -> "CodigoMorse":
        pasta = Path(pasta)
        t = json.loads((pasta / "tabela.json").read_text(encoding="utf-8"))
        meta = json.loads((pasta / "metadados.json").read_text(encoding="utf-8"))
        codigos = [tuple(int(s[k:k + 4], 16) for k in range(0, len(s), 4)) for s in t["codigos"]]
        return cls(t["inventario"], codigos, t["A"], t["topo"], t["variante"], meta)

    def metadados(self) -> dict:
        return {"implementacao": "externo", "vocab_size": self.vocab_size, **{k: v for k, v in self._meta.items() if k not in ("modulo", "classe")}}


def main():
    from slm.corpus import ler_documentos
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="etapa", required=True)
    s = sub.add_parser("construir")
    s.add_argument("--A", type=int, default=256)
    s.add_argument("--topo", type=int, default=4096)
    s.add_argument("--variante", default="prefixo")
    s.add_argument("--saida", required=True)
    s = sub.add_parser("verificar")
    s.add_argument("pasta")
    a = ap.parse_args()
    if a.etapa == "construir":
        tok = CodigoMorse.construir([d["texto"] for d in ler_documentos("A", "treino")], a.A, a.topo, a.variante)
        tok.salvar(Path(a.saida))
        print(json.dumps(tok.metadados(), indent=2))
        return
    from slm.tokenizador import carregar_tokenizador
    tok = carregar_tokenizador(a.pasta)
    rel = {}
    for corpus in ("A", "B"):
        for part in ("treino", "validacao", "teste"):
            docs = [d["texto"] for d in ler_documentos(corpus, part)]
            B = sum(len(t.encode("utf-8")) for t in docs)
            n, falhas = 0, 0
            for t in docs:
                ids = tok.encode(t)
                n += len(ids)
                falhas += tok.decode(ids) != t.encode("utf-8")
                assert all(N_ESP <= i < tok.vocab_size for i in ids)
            rel[f"{corpus}/{part}"] = {"documentos": len(docs), "falhas_ida_e_volta": falhas, "simbolos_por_byte": round(n / B, 4)}
    print(json.dumps(rel, indent=1))


if __name__ == "__main__":
    main()

"""Tokenizer contract, didactic byte-level BPE and library adapter (box 2).

Contract (docs/CONTRATO_DADO.md, section 2): encode(str) -> ids, decode(ids) -> bytes,
vocab_size, especiais, salvar/carregar, metadados. Special ids 0..3 are reserved and never
produced from text. Base byte symbols take ids 4..259, ordered by the character that the
GPT-2 bytes_to_unicode map assigns to each byte, and every merge takes the next id. Both
the didactic BPE and the library follow this layout, so equal merges imply equal ids.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable

import regex

ESPECIAIS = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[MASK]": 3}
N_ESP = len(ESPECIAIS)
GPT2_REGEX = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


@runtime_checkable
class Tokenizador(Protocol):
    vocab_size: int
    especiais: dict

    def encode(self, texto: str) -> list[int]: ...
    def decode(self, ids: Iterable[int]) -> bytes: ...
    def salvar(self, pasta: Path) -> None: ...
    def metadados(self) -> dict: ...


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 reversible map from each byte to a printable character."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


B2U = bytes_to_unicode()
U2B = {c: b for b, c in B2U.items()}
# base symbols in the order the library assigns them
ORDEM_BYTES = sorted(range(256), key=lambda b: ord(B2U[b]))


def sha256_textos(textos: Iterable[str]) -> str:
    h = hashlib.sha256()
    for t in textos:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class TokenizadorBytes:
    """Trivial tokenizer: one id per byte. Proves the contract does not assume BPE (test U6)."""

    especiais = dict(ESPECIAIS)
    vocab_size = N_ESP + 256

    def encode(self, texto: str) -> list[int]:
        return [N_ESP + b for b in texto.encode("utf-8")]

    def decode(self, ids: Iterable[int]) -> bytes:
        return bytes(i - N_ESP for i in ids if i >= N_ESP)

    def salvar(self, pasta: Path) -> None:
        pasta = Path(pasta)
        pasta.mkdir(parents=True, exist_ok=True)
        (pasta / "metadados.json").write_text(json.dumps(self.metadados(), indent=2), encoding="utf-8")

    @classmethod
    def carregar(cls, pasta: Path) -> "TokenizadorBytes":
        return cls()

    def metadados(self) -> dict:
        return {"implementacao": "bytes", "vocab_size": self.vocab_size}


class BPEDidatico:
    """Byte-level BPE written for teaching.

    Training: count adjacent pairs inside pre-tokens, weighted by pre-token frequency, and merge
    the most frequent pair. Tie-break convention: highest count, then the smallest (left id,
    right id). Naive cost O(N * M) for N symbols and M merges, adequate for the notebook sample.
    """

    especiais = dict(ESPECIAIS)

    def __init__(self, merges: list[tuple[bytes, bytes]] | None = None, regex_pt: str = GPT2_REGEX):
        self.regex = regex_pt
        self._pat = regex.compile(regex_pt)
        self.merges: list[tuple[bytes, bytes]] = []
        self.vocab: dict[bytes, int] = {bytes([b]): N_ESP + i for i, b in enumerate(ORDEM_BYTES)}
        self._meta: dict = {}
        self._rank: dict[tuple[bytes, bytes], int] = {}
        for a, b in merges or []:
            self._adicionar(a, b)
        self._cache: dict[str, list[int]] = {}

    def _adicionar(self, a: bytes, b: bytes) -> None:
        self._rank[(a, b)] = len(self.merges)
        self.merges.append((a, b))
        self.vocab[a + b] = len(self.vocab) + N_ESP

    @property
    def vocab_size(self) -> int:
        return N_ESP + len(self.vocab)

    def treinar(self, textos: Iterable[str], vocab_size: int) -> "BPEDidatico":
        textos = list(textos)
        palavras = collections.Counter(
            tuple(bytes([b]) for b in w.encode("utf-8")) for t in textos for w in self._pat.findall(t))
        while self.vocab_size < vocab_size:
            stats = collections.Counter()
            for w, c in palavras.items():
                for p in zip(w, w[1:]):
                    stats[p] += c
            if not stats:
                break
            ids = self.vocab
            best = max(stats, key=lambda p: (stats[p], -ids[p[0]], -ids[p[1]]))
            self._adicionar(*best)
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
        self._cache.clear()
        self._meta = {"sha256_treino": sha256_textos(textos)}
        return self

    def _encode_palavra(self, w: str) -> list[int]:
        if w in self._cache:
            return self._cache[w]
        parts = [bytes([b]) for b in w.encode("utf-8")]
        while len(parts) > 1:
            ranks = [self._rank.get(p, len(self.merges)) for p in zip(parts, parts[1:])]
            r = min(ranks)
            if r == len(self.merges):
                break
            a, b = self.merges[r]
            out, i = [], 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == a and parts[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(parts[i])
                    i += 1
            parts = out
        ids = [self.vocab[p] for p in parts]
        self._cache[w] = ids
        return ids

    def encode(self, texto: str) -> list[int]:
        return [i for w in self._pat.findall(texto) for i in self._encode_palavra(w)]

    def decode(self, ids: Iterable[int]) -> bytes:
        inv = getattr(self, "_inv", None)
        if inv is None or len(inv) != len(self.vocab):
            inv = self._inv = {i: s for s, i in self.vocab.items()}
        return b"".join(inv[i] for i in ids if i >= N_ESP)

    def salvar(self, pasta: Path) -> None:
        pasta = Path(pasta)
        pasta.mkdir(parents=True, exist_ok=True)
        dados = {"regex": self.regex, "merges": [[a.hex(), b.hex()] for a, b in self.merges]}
        (pasta / "bpe_didatico.json").write_text(json.dumps(dados), encoding="utf-8")
        (pasta / "metadados.json").write_text(json.dumps(self.metadados(), indent=2), encoding="utf-8")

    @classmethod
    def carregar(cls, pasta: Path) -> "BPEDidatico":
        dados = json.loads((Path(pasta) / "bpe_didatico.json").read_text(encoding="utf-8"))
        return cls([(bytes.fromhex(a), bytes.fromhex(b)) for a, b in dados["merges"]], dados["regex"])

    def metadados(self) -> dict:
        return {"implementacao": "didatico", "vocab_size": self.vocab_size, "regex": self.regex, **self._meta}


class BPEBiblioteca:
    """Byte-level BPE trained by the Hugging Face tokenizers library, same layout as BPEDidatico."""

    especiais = dict(ESPECIAIS)

    def __init__(self, tok=None, regex_pt: str = GPT2_REGEX, meta: dict | None = None):
        self.tok = tok
        self.regex = regex_pt
        self._meta = meta or {}

    @staticmethod
    def _novo(regex_pt: str):
        from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(regex_pt), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
        tok.decoder = decoders.ByteLevel()
        return tok

    def treinar(self, textos: Iterable[str], vocab_size: int) -> "BPEBiblioteca":
        import tokenizers
        from tokenizers import pre_tokenizers, trainers
        textos = list(textos)
        self.tok = self._novo(self.regex)
        tr = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=0, show_progress=False,
                                 special_tokens=list(ESPECIAIS), initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        self.tok.train_from_iterator(textos, tr)
        self.tok.encode_special_tokens = True
        self._meta = {"versao_biblioteca": tokenizers.__version__, "sha256_treino": sha256_textos(textos)}
        return self

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, texto: str) -> list[int]:
        return self.tok.encode(texto, add_special_tokens=False).ids

    def decode(self, ids: Iterable[int]) -> bytes:
        return b"".join(bytes(U2B[c] for c in self.tok.id_to_token(i)) for i in ids if i >= N_ESP)

    def merges(self) -> list[tuple[bytes, bytes]]:
        raw = json.loads(self.tok.to_str())["model"]["merges"]
        pares = (r.split(" ") if isinstance(r, str) else r for r in raw)
        return [tuple(bytes(U2B[c] for c in x) for x in p) for p in pares]

    def salvar(self, pasta: Path) -> None:
        pasta = Path(pasta)
        pasta.mkdir(parents=True, exist_ok=True)
        self.tok.save(str(pasta / "tokenizer.json"))
        meta = self.metadados()
        meta["sha256_tokenizer_json"] = hashlib.sha256((pasta / "tokenizer.json").read_bytes()).hexdigest()
        (pasta / "metadados.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def carregar(cls, pasta: Path) -> "BPEBiblioteca":
        from tokenizers import Tokenizer
        pasta = Path(pasta)
        meta = json.loads((pasta / "metadados.json").read_text(encoding="utf-8"))
        tok = Tokenizer.from_file(str(pasta / "tokenizer.json"))
        tok.encode_special_tokens = True
        return cls(tok, meta.get("regex", GPT2_REGEX), meta)

    def metadados(self) -> dict:
        return {"implementacao": "biblioteca", "vocab_size": self.vocab_size, "regex": self.regex, **self._meta}


def fracao_ausente(tok: Tokenizador, retido: Iterable[str]) -> float:
    """Fraction of learned tokens (ids above the byte alphabet) absent from the held-out text."""
    vistos = {i for t in retido for i in tok.encode(t)}
    aprendidos = range(N_ESP + 256, tok.vocab_size)
    return sum(i not in vistos for i in aprendidos) / max(1, len(aprendidos))


def escolher_vocabulario(treino: list[str], retido: list[str], grade=(1024, 2048, 4096, 8192),
                         limiar: float = 0.10, fabrica=BPEBiblioteca) -> tuple[int, dict]:
    """Largest V in the grid with at most `limiar` of learned tokens absent from the held-out text."""
    medidas = {V: fracao_ausente(fabrica().treinar(treino, V), retido) for V in grade}
    validos = [V for V, f in medidas.items() if f <= limiar]
    return (max(validos) if validos else min(grade)), medidas


def carregar_tokenizador(pasta: Path | str) -> Tokenizador:
    """Load any tokenizer that honours the contract, chosen by `implementacao` in metadados.json.

    "externo" loads the class `classe` from the Python file `modulo`, given relative to the
    tokenizer folder, so a tokenizer kept outside the package plugs into training unchanged.
    """
    pasta = Path(pasta)
    meta = json.loads((pasta / "metadados.json").read_text(encoding="utf-8"))
    impl = meta.get("implementacao")
    if impl == "externo":
        modulo = (pasta / meta["modulo"]).resolve()
        spec = importlib.util.spec_from_file_location(modulo.stem, modulo)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, meta["classe"]).carregar(pasta)
    classes = {"biblioteca": BPEBiblioteca, "didatico": BPEDidatico, "bytes": TokenizadorBytes}
    if impl not in classes:
        raise ValueError(f"implementacao desconhecida em {pasta}: {impl}")
    return classes[impl].carregar(pasta)


def conferir_contrato(tok: Tokenizador, textos: list[str], pasta: Path | str | None = None) -> dict[str, bool]:
    """Contract checks U1 to U5 on the given texts, for any tokenizer (U6 is that bytes, BPE and any
    other implementation pass the same checks). With `pasta`, U2 also saves and reloads the tokenizer
    through carregar_tokenizador."""
    ids = [tok.encode(t) for t in textos]
    sonda = "inicio [PAD][BOS][EOS][MASK] fim"
    res = {"U1 ida e volta exata": all(tok.decode(i) == t.encode("utf-8") for i, t in zip(ids, textos)),
           "U2 determinismo": all(tok.encode(t) == i for t, i in zip(textos, ids)),
           "U3 ids dentro do vocabulário": all(0 <= x < tok.vocab_size for i in ids for x in i),
           "U4 especiais nunca gerados": all(x >= N_ESP for i in ids + [tok.encode(sonda)] for x in i),
           "U5 decode de id isolado": all(isinstance(tok.decode([x]), bytes) for x in range(tok.vocab_size))}
    if pasta is not None:
        tok.salvar(Path(pasta))
        outro = carregar_tokenizador(pasta)
        res["U2 persistência"] = all(outro.encode(t) == i for t, i in zip(textos, ids))
    return res


def hash_tokenizador(tok: Tokenizador) -> str:
    """Content hash recorded by the tokenizer in its metadata, empty when it has none."""
    meta = tok.metadados()
    return meta.get("sha256_tokenizer_json") or meta.get("sha256", "")


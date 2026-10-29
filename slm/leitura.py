"""Lexicon rate, copy filter, blind batches and agreement (box 10).

Definitions follow docs/SPEC.md, section 10, and docs/CONTRATO_DADO.md, section 6.
"""

from __future__ import annotations

import csv
import itertools
import json
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from slm.estatistica import alfa_krippendorff, clopper_pearson, clopper_pearson_inferior

NIVEIS = ("N2", "N3", "N4")
_CANDIDATO = re.compile(r"[^\W\d_]+")


# --------------------------------------------------------------------------
# Words and level N1
# --------------------------------------------------------------------------

def palavras(texto: str) -> list[str]:
    """Maximal runs of Unicode letters (str.isalpha), lowercased, after NFC."""
    texto = unicodedata.normalize("NFC", texto)
    out = []
    for m in _CANDIDATO.findall(texto):
        if m.isalpha():
            out.append(m.lower())
        else:
            # rare non-decimal numerics (e.g. superscripts) inside a \w run
            out.extend("".join(g).lower() for k, g in itertools.groupby(m, str.isalpha) if k)
    return out


def lexico(textos: Iterable[str]) -> frozenset[str]:
    """Set of words of the training partition."""
    return frozenset(itertools.chain.from_iterable(palavras(t) for t in textos))


def taxa_lexico(texto: str, lex: frozenset[str]) -> float:
    """Fraction of the words of `texto` present in `lex`; nan without words."""
    ps = palavras(texto)
    return sum(p in lex for p in ps) / len(ps) if ps else float("nan")


def trigramas(textos: Iterable[str]) -> frozenset[tuple[str, str, str]]:
    """Set of word trigrams of the training partition, within each text."""
    out = set()
    for t in textos:
        ps = palavras(t)
        out.update(zip(ps, ps[1:], ps[2:]))
    return frozenset(out)


def taxa_trigramas(texto: str, tri: frozenset[tuple[str, str, str]]) -> float:
    """Fraction of the word trigrams of `texto` attested in `tri`; nan with fewer than 3 words."""
    ps = palavras(texto)
    ts = list(zip(ps, ps[1:], ps[2:]))
    return sum(t in tri for t in ts) / len(ts) if ts else float("nan")


@dataclass(frozen=True)
class ResultadoN1:
    taxa_amostras: float
    taxa_humanos: float
    diferenca: float  # taxa_amostras - taxa_humanos
    atingido: bool


def nivel_n1(taxas_amostras, taxas_humanos, limite: float = 0.05) -> ResultadoN1:
    """N1 holds when |mean(samples) - mean(human)| <= limite (5 pp is a convention)."""
    a = float(np.nanmean(np.asarray(taxas_amostras, dtype=np.float64)))
    h = float(np.nanmean(np.asarray(taxas_humanos, dtype=np.float64)))
    return ResultadoN1(a, h, a - h, bool(abs(a - h) <= limite + 1e-12))


# --------------------------------------------------------------------------
# Copy filter: longest contiguous span shared with the training partition
# --------------------------------------------------------------------------

_P1, _P2 = 2_147_483_647, 2_147_483_629  # primes below 2**31
_B1, _B2 = 1_000_003, 999_983


def _potencias(base: int, p: int, n: int) -> np.ndarray:
    """base**i mod p for i in [0, n], by doubling (log n vectorized steps)."""
    out = np.ones(n + 1, dtype=np.int64)
    k, passo = 1, base % p
    while k <= n:
        fim = min(2 * k, n + 1)
        out[k:fim] = out[: fim - k] * passo % p
        passo = passo * passo % p
        k *= 2
    return out


class IndiceTreino:
    """Index of the training word sequence for longest-copied-span queries.

    Method: double polynomial rolling hash (two primes near 2**31, packed in
    int64) with binary search on the span length. For one length l, all
    training l-grams are hashed in O(N) vectorized time and sorted in
    O(N log N); the sorted array is cached per l. A query of m words costs
    O(m log m log N) once the needed lengths are cached. Spans never cross
    document boundaries. Collision probability per comparison is about
    2**-61, so a false match is negligible for corpora of this size.
    """

    def __init__(self, textos_treino: Iterable[str], cache_max: int = 64):
        self.vocab: dict[str, int] = {}
        ids, docs = [], []
        for j, t in enumerate(textos_treino):
            ps = palavras(t)
            ids.extend(self.vocab.setdefault(p, len(self.vocab) + 1) for p in ps)
            docs.extend([j] * len(ps))
        self._ids = np.asarray(ids, dtype=np.int64)
        self._docs = np.asarray(docs, dtype=np.int64)
        self._prefixos = self._prefixo(self._ids)
        self._cache: dict[int, np.ndarray] = {}
        self._cache_max = cache_max
        self._inv = (_potencias(pow(_B1, -1, _P1), _P1, self._ids.size),
                     _potencias(pow(_B2, -1, _P2), _P2, self._ids.size))

    @staticmethod
    def _prefixo(ids: np.ndarray):
        out = []
        for b, p in ((_B1, _P1), (_B2, _P2)):
            termos = ids % p * _potencias(b, p, ids.size)[: ids.size] % p
            out.append(np.concatenate(([0], np.cumsum(termos) % p)))  # terms < 2**31, so int64 cumsum is exact for N < 2**32
        return out

    @staticmethod
    def _janelas(prefixos, inv, ell: int, inicios: np.ndarray) -> np.ndarray:
        h = []
        for pre, iv, p in zip(prefixos, inv, (_P1, _P2)):
            dif = (pre[inicios + ell] - pre[inicios]) % p
            h.append(dif * iv[inicios] % p)
        return h[0] * _P1 + h[1]

    def _tabela(self, ell: int) -> np.ndarray:
        tab = self._cache.get(ell)
        if tab is None:
            n = self._ids.size
            if ell > n:
                tab = np.empty(0, dtype=np.int64)
            else:
                ini = np.arange(n - ell + 1)
                ini = ini[self._docs[ini] == self._docs[ini + ell - 1]]
                tab = np.unique(self._janelas(self._prefixos, self._inv, ell, ini))
            if len(self._cache) >= self._cache_max:
                self._cache.pop(next(iter(self._cache)))
            self._cache[ell] = tab
        return tab

    def _ids_consulta(self, ps: Sequence[str]) -> np.ndarray:
        # unknown words get fresh ids, so no gram containing them can match
        novo = itertools.count(len(self.vocab) + 1)
        return np.fromiter((self.vocab.get(p) or next(novo) for p in ps), dtype=np.int64, count=len(ps))

    def maior_trecho(self, texto: str) -> int:
        """Length, in words, of the longest contiguous span of `texto` found in training."""
        return int(self.maiores_trechos([texto])[0])

    def maiores_trechos(self, textos: Sequence[str]) -> np.ndarray:
        """Batch version of `maior_trecho`; queries sharing a length share one table."""
        consultas = [self._ids_consulta(palavras(t)) for t in textos]
        pre = [self._prefixo(q) for q in consultas]
        m_max = max((q.size for q in consultas), default=0)
        inv = (_potencias(pow(_B1, -1, _P1), _P1, m_max), _potencias(pow(_B2, -1, _P2), _P2, m_max))
        lo = np.zeros(len(consultas), dtype=np.int64)
        hi = np.array([q.size for q in consultas], dtype=np.int64)
        while np.any(lo < hi):
            meio = (lo + hi + 1) // 2
            ativos = np.flatnonzero(lo < hi)
            for ell in np.unique(meio[ativos]):
                tab = self._tabela(int(ell))
                for i in ativos[meio[ativos] == ell]:
                    hs = self._janelas(pre[i], inv, int(ell), np.arange(consultas[i].size - ell + 1))
                    pos = np.searchsorted(tab, hs)
                    achou = bool(np.any(tab[np.minimum(pos, tab.size - 1)] == hs)) if tab.size else False
                    if achou:
                        lo[i] = ell
                    else:
                        hi[i] = ell - 1
        return lo


def limiar_copia(c_humanos, percentil: float = 99.0, metodo: str = "linear") -> float:
    """k = percentile of the longest copied span among human test excerpts."""
    return float(np.percentile(np.asarray(c_humanos, dtype=np.float64), percentil, method=metodo))


@dataclass(frozen=True)
class FiltroCopia:
    descartadas: np.ndarray  # bool per sample, c > k
    n_descartadas: int
    n: int
    taxa: float
    intervalo: tuple[float, float]  # two-sided Clopper-Pearson


def filtro_copia(c_amostras, k: float, confianca: float = 0.95) -> FiltroCopia:
    """Discard samples with c > k and report the discard rate with its exact interval."""
    c = np.asarray(c_amostras)
    desc = c > k
    x, n = int(desc.sum()), int(desc.size)
    return FiltroCopia(desc, x, n, x / n if n else float("nan"), clopper_pearson(x, n, confianca))


# --------------------------------------------------------------------------
# Blind batches
# --------------------------------------------------------------------------

def embaralhar(texto: str, semente: int) -> str:
    """Shuffled-word control: random permutation of the whitespace tokens."""
    toks = texto.split()
    random.Random(semente).shuffle(toks)
    return " ".join(toks)


def montar_lote(
    itens_modelo: Sequence[Mapping],
    pares_controle: Sequence[tuple[str, str]],
    semente: int,
) -> tuple[list[dict], dict[str, dict]]:
    """Build a blind batch in random order.

    `itens_modelo` has dicts with `texto`, `modo`, `ponto` and `semente`
    (extra keys such as `amostra_id` go to the key). `pares_controle` lists
    (human, shuffled) text pairs; pair j yields one `humano` and one
    `embaralhado` item with `par` = j. Item ids are assigned after shuffling,
    so neither id nor position carries the label.
    """
    entradas = []
    for it in itens_modelo:
        rotulo = {"tipo": "modelo", **{k: v for k, v in it.items() if k != "texto"}}
        entradas.append((it["texto"], rotulo))
    for j, (hum, emb) in enumerate(pares_controle):
        entradas.append((hum, {"tipo": "humano", "par": j}))
        entradas.append((emb, {"tipo": "embaralhado", "par": j}))
    random.Random(semente).shuffle(entradas)
    largura = max(3, len(str(len(entradas))))
    linhas, chave = [], {}
    for i, (texto, rotulo) in enumerate(entradas, 1):
        item_id = f"i{i:0{largura}d}"
        linhas.append({"item_id": item_id, "texto": unicodedata.normalize("NFC", texto)})
        chave[item_id] = rotulo
    return linhas, chave


def gravar_lote(pasta: Path | str, n: int, linhas: Sequence[Mapping], chave: Mapping) -> tuple[Path, Path]:
    """Write `lote_<n>.csv` (item_id, texto) and `chave_<n>.json`."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    p_lote, p_chave = pasta / f"lote_{n}.csv", pasta / f"chave_{n}.json"
    with open(p_lote, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "texto"])
        w.writeheader()
        w.writerows({"item_id": l["item_id"], "texto": l["texto"]} for l in linhas)
    with open(p_chave, "w", encoding="utf-8") as f:
        json.dump(dict(chave), f, ensure_ascii=False, indent=2, sort_keys=True)
    return p_lote, p_chave


def ler_lote(caminho: Path | str) -> list[dict]:
    with open(caminho, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def ler_chave(caminho: Path | str) -> dict[str, dict]:
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def ler_respostas(caminho: Path | str) -> dict[str, dict[str, int | None]]:
    """Read `respostas_<avaliador>_<n>.csv`; empty cells become None."""
    out = {}
    with open(caminho, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        faltam = {"item_id", *NIVEIS} - set(r.fieldnames or ())
        if faltam:
            raise ValueError(f"colunas ausentes: {sorted(faltam)}")
        for linha in r:
            notas = {}
            for nv in NIVEIS:
                v = (linha[nv] or "").strip()
                if v == "":
                    notas[nv] = None
                    continue
                nota = int(v)
                if nota not in (0, 1, 2):
                    raise ValueError(f"nota fora de 0 a 2 em {linha['item_id']}: {v}")
                notas[nv] = nota
            out[linha["item_id"]] = notas
    return out


# --------------------------------------------------------------------------
# Levels N2 to N4, rater filter and agreement
# --------------------------------------------------------------------------

def matriz_notas(
    respostas: Mapping[str, Mapping[str, Mapping[str, int | None]]],
    itens: Sequence[str],
    nivel: str,
    avaliadores: Sequence[str] | None = None,
) -> np.ndarray:
    """(raters, items) score matrix for one level, nan where missing."""
    avs = list(respostas) if avaliadores is None else list(avaliadores)
    m = np.full((len(avs), len(itens)), np.nan)
    for a, av in enumerate(avs):
        r = respostas[av]
        for i, it in enumerate(itens):
            v = r.get(it, {}).get(nivel)
            if v is not None:
                m[a, i] = v
    return m


@dataclass(frozen=True)
class ResultadoNivel:
    sucessos: int
    n: int
    atingido: bool
    limite_inferior: float  # one-sided Clopper-Pearson


def nivel_atingido(notas, minimo_itens: int = 10, nota_minima: float = 2, confianca: float = 0.95) -> ResultadoNivel:
    """A level counts when at least `minimo_itens` items have rater median >= `nota_minima`.

    `notas` has shape (raters, items); items without any score count as failures.
    """
    notas = np.asarray(notas, dtype=np.float64)
    tem = ~np.all(np.isnan(notas), axis=0)
    med = np.full(notas.shape[1], -np.inf)
    med[tem] = np.nanmedian(notas[:, tem], axis=0)
    x, n = int(np.sum(med >= nota_minima)), int(notas.shape[1])
    return ResultadoNivel(x, n, x >= minimo_itens, clopper_pearson_inferior(x, n, confianca))


def itens_modelo(chave: Mapping[str, Mapping], **filtro) -> list[str]:
    """Model item ids of a key, optionally filtered (e.g. modo=..., ponto=...)."""
    return sorted(
        i for i, r in chave.items()
        if r.get("tipo") == "modelo" and all(r.get(k) == v for k, v in filtro.items())
    )


@dataclass(frozen=True)
class FiltroAvaliador:
    acertos: int
    pares: int
    aprovado: bool


def filtro_avaliador(
    respostas: Mapping[str, Mapping[str, int | None]],
    chave: Mapping[str, Mapping],
    minimo: int = 18,
    niveis: Sequence[str] = NIVEIS,
) -> FiltroAvaliador:
    """Rater passes when the human excerpt scores strictly above its shuffled pair in >= `minimo` pairs.

    The comparison uses the sum of the scores over `niveis`; a tie or a
    missing score counts as a failure.
    """
    pares: dict[int, dict[str, str]] = {}
    for item_id, r in chave.items():
        if r.get("tipo") in ("humano", "embaralhado"):
            pares.setdefault(int(r["par"]), {})[r["tipo"]] = item_id

    def total(item_id):
        notas = [respostas.get(item_id, {}).get(nv) for nv in niveis]
        return None if any(v is None for v in notas) else sum(notas)

    acertos = 0
    for par in pares.values():
        h, e = total(par.get("humano")), total(par.get("embaralhado"))
        acertos += h is not None and e is not None and h > e
    return FiltroAvaliador(acertos, len(pares), acertos >= minimo)


@dataclass(frozen=True)
class Concordancia:
    alfa: float
    aprovada: bool


def concordancia(
    respostas: Mapping[str, Mapping[str, Mapping[str, int | None]]],
    chave: Mapping[str, Mapping],
    nivel: str,
    avaliadores: Sequence[str] | None = None,
    limiar: float = 0.667,
) -> Concordancia:
    """Ordinal Krippendorff's alpha over model items only, for one level."""
    alfa = alfa_krippendorff(matriz_notas(respostas, itens_modelo(chave), nivel, avaliadores), "ordinal")
    return Concordancia(alfa, bool(np.isfinite(alfa) and alfa >= limiar))

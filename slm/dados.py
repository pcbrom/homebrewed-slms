"""Domain adapters, cleaning, segmentation and split for corpora A and B (box 1).

Pure functions, standard library only. The build scripts in ``scripts/`` do the
network and file work and call these functions.
"""

from __future__ import annotations

import hashlib
import html
import re
import struct
import unicodedata
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser

# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def sha256_bytes(dados: bytes) -> str:
    return hashlib.sha256(dados).hexdigest()


def sha256_texto(texto: str) -> str:
    return sha256_bytes(texto.encode("utf-8"))


def n_bytes(texto: str) -> int:
    return len(texto.encode("utf-8"))


def normalizar(texto: str) -> str:
    """CRLF and CR to LF, then NFC."""
    return unicodedata.normalize("NFC", texto.replace("\r\n", "\n").replace("\r", "\n"))


# ---------------------------------------------------------------------------
# Corpus A: Project Gutenberg
# ---------------------------------------------------------------------------

_RE_START = re.compile(r"^\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*\*\*\*\s*$", re.M)
_RE_END = re.compile(r"^\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*\*\*\*\s*$", re.M)


def cortar_gutenberg(bruto: str) -> str:
    """Keep only the text strictly between the START and END marker lines."""
    texto = normalizar(bruto)
    ini = _RE_START.search(texto)
    fim = _RE_END.search(texto)
    if ini is None or fim is None or fim.start() <= ini.end():
        raise ValueError("marcas START/END do Gutenberg ausentes ou fora de ordem")
    return texto[ini.end():fim.start()]


_ROMANOS = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_RE_ROMANO = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")


def romano_para_int(s: str) -> int:
    s = s.strip().upper()
    if not s or not _RE_ROMANO.match(s):
        raise ValueError(f"algarismo romano invalido: {s!r}")
    total = 0
    for a, b in zip(s, s[1:] + " "):
        v = _ROMANOS[a]
        total += -v if b != " " and _ROMANOS[b] > v else v
    return total


_MESES = "janeiro|fevereiro|março|marco|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro"

# Heading patterns. Each one matches a whole stripped line.
_RE_CAB_ROMANO = re.compile(r"^([IVXLCDM]+)\.?$")
# CAPIULO covers a misprint of the source (Braz Cubas, chapter CXXIV).
_RE_CAB_CAPITULO = re.compile(r"^CAP[IÍ]T?ULO\s+([IVXLCDM]+|PRIMEIRO|UNICO|ÚNICO)\.?$")
_RE_CAB_DATA = re.compile(rf"^(\d{{1,2}})\s+(?:d[eo]\s+)?({_MESES})\b.{{0,50}}$|^Sem data\.?$", re.I)
_RE_ANO = re.compile(r"^1[89]\d\d\.?$")
_RE_FIM_OBRA = re.compile(r"^(FIM|FIN)\.?$|^[IÍ]NDICE$", re.I)
_RE_FIM_CONTO = re.compile(r"^FIM D[AEO]S?\s")
_RE_SEPARADOR = re.compile(r"^(\*\s*){3,}$")
# Bracketed notes: footnote markers such as [1] and transcriber notes such as [signature].
_RE_NOTA = re.compile(r"\[[^\]\n]*\]")
# One or two asterisks mark transcriber corrections; three or more hide a name in the original.
_RE_ASTERISCO = re.compile(r"(?<!\*)\*{1,2}(?!\*)")
_RE_HIFENS = re.compile(r"-{2,}")


@dataclass(frozen=True)
class Obra:
    """Per-work segmentation rule for corpus A."""

    ebook: str
    slug: str
    titulo: str
    regra: str  # romano, capitulo, data or conto
    inicio: str | None = None  # regex of the first body line, when the rule alone is ambiguous


OBRAS_A: tuple[Obra, ...] = (
    Obra("55752", "dom_casmurro", "Dom Casmurro", "romano"),
    Obra("54829", "braz_cubas", "Memorias posthumas de Braz Cubas", "capitulo"),
    Obra("55682", "quincas_borba", "Quincas Borba", "capitulo"),
    Obra("57001", "papeis_avulsos", "Papeis avulsos", "conto", r"^O ALIENISTA$"),
    Obra("33056", "historias_sem_data", "Historias sem data", "conto", r"^A EGREJA DO DIABO$"),
    Obra("53101", "mao_e_luva", "A mao e a luva", "romano"),
    Obra("67162", "helena", "Helena", "capitulo"),
    Obra("56737", "esau_e_jacob", "Esau e Jacob", "capitulo"),
    Obra("55797", "memorial_de_ayres", "Memorial de Ayres", "data"),
    Obra("67780", "yaya_garcia", "Yaya Garcia", "romano"),
)

EXCLUIDAS_A: dict[str, str] = {
    "61653": "poesia (Poesias completas), fora do corpus de prosa",
    "67935": "contem soneto e duas comedias (Reliquias de casa velha), fora do corpus de prosa",
}


def _tem_minuscula(s: str) -> bool:
    return any(c.islower() for c in s)


def _e_numeracao(linha: str) -> bool:
    s = linha.strip()
    m = _RE_CAB_ROMANO.match(s)
    return bool((m and _RE_ROMANO.match(m.group(1))) or _RE_CAB_CAPITULO.match(s))


def _e_titulo_conto(linhas: list[str], i: int) -> bool:
    s = linhas[i].strip()
    if not s or _tem_minuscula(s) or sum(c.isalpha() for c in s) < 2 or _RE_FIM_CONTO.match(s):
        return False
    brancos = 0
    j = i - 1
    while j >= 0 and not linhas[j].strip():
        brancos += 1
        j -= 1
    return brancos >= 4 or j < 0


def _e_cabecalho(linhas: list[str], i: int, regra: str) -> bool:
    s = linhas[i].strip()
    anterior_branca = i == 0 or not linhas[i - 1].strip()
    if regra == "romano":
        m = _RE_CAB_ROMANO.match(s)
        return bool(m and _RE_ROMANO.match(m.group(1)) and anterior_branca)
    if regra == "capitulo":
        return bool(_RE_CAB_CAPITULO.match(s) and anterior_branca)
    if regra == "data":
        # Dates inside letters are indented, entry dates start at column 0.
        return bool(_RE_CAB_DATA.match(s) and linhas[i] == linhas[i].lstrip() and anterior_branca)
    if regra == "conto":
        return _e_titulo_conto(linhas, i)
    raise ValueError(f"regra desconhecida: {regra}")


def segmentar(texto: str, regra: str, inicio: str | None = None) -> list[tuple[str, list[str]]]:
    """Split the body of a work into (heading, lines) units.

    The front matter before the first heading (or before ``inicio``) is dropped,
    leading units with no lowercase letter (table of contents, half-title) are
    dropped, and the body ends at the first end marker (FIM, FIN, INDICE).
    """
    linhas = texto.split("\n")
    i0 = 0
    if inicio is not None:
        rx = re.compile(inicio)
        i0 = next((i for i, l in enumerate(linhas) if rx.match(l.strip())), None)
        if i0 is None:
            raise ValueError(f"inicio {inicio!r} nao encontrado")
    unidades: list[tuple[str, list[str]]] = []
    atual: tuple[str, list[str]] | None = None
    for i in range(i0, len(linhas)):
        s = linhas[i].strip()
        if atual is not None and _RE_FIM_OBRA.match(s) and any(_tem_minuscula(l) for l in atual[1]):
            break
        if _e_cabecalho(linhas, i, regra) or (i == i0 and inicio is not None):
            atual = (s, [])
            unidades.append(atual)
        elif atual is not None:
            atual[1].append(linhas[i])
    while unidades and not any(_tem_minuscula(l) for l in unidades[0][1]):
        unidades.pop(0)
    return unidades


def _juntar_paragrafo(linhas: list[str]) -> str:
    # A paragraph whose lines are all indented is verse or a set-off quote: keep its line breaks.
    if len(linhas) > 1 and all(l.startswith("    ") for l in linhas):
        return "\n".join(re.sub(r"[ \t]+", " ", l.strip()) for l in linhas)
    return re.sub(r"[ \t]+", " ", " ".join(l.strip() for l in linhas))


def limpar_texto_gutenberg(linhas: list[str], regra: str) -> str:
    """Clean the lines of one unit.

    Drops numbering lines, scene separators, end-of-story lines, bracketed notes
    and transcriber correction asterisks, joins the lines of each paragraph with
    paragraphs separated by one blank line, removes italics underscores and
    writes every run of two or more hyphens (the dash of the transcription) as
    exactly ``--``.
    """
    if regra == "data":
        linhas = [l for l in linhas if not _RE_ANO.match(l.strip())]
    limpas: list[str] = []
    brancos = 0
    for l in linhas:
        s = l.strip()
        if not s:
            brancos += 1
            limpas.append("")
            continue
        if _RE_SEPARADOR.match(s) or (regra == "conto" and _RE_FIM_CONTO.match(s)):
            limpas.append("")
        elif brancos >= 2 and _e_numeracao(s):
            limpas.append("")
        else:
            limpas.append(l.rstrip())
        brancos = 0
    paragrafos: list[str] = []
    bloco: list[str] = []
    for l in limpas + [""]:
        if l.strip():
            bloco.append(l)
        elif bloco:
            paragrafos.append(_juntar_paragrafo(bloco))
            bloco = []
    texto = "\n\n".join(paragrafos)
    texto = _RE_ASTERISCO.sub("", _RE_NOTA.sub("", texto)).replace("_", "")
    texto = re.sub(r"[ \t]{2,}", " ", _RE_HIFENS.sub("--", texto))
    return unicodedata.normalize("NFC", texto).strip()


def limpar_obra(bruto: str, obra: Obra) -> list[tuple[str, str]]:
    """Full cleaning of one Gutenberg file: returns (heading, clean text) per unit.

    For ``data`` and ``conto`` the heading line is part of the text (date of the
    entry, title of the story). For ``romano`` and ``capitulo`` it is numbering
    and is dropped, while the chapter title on the next line, if any, is kept.
    """
    corpo = cortar_gutenberg(bruto)
    saida = []
    for cab, linhas in segmentar(corpo, obra.regra, obra.inicio):
        if obra.regra in ("data", "conto"):
            linhas = [cab] + linhas
        texto = limpar_texto_gutenberg(linhas, obra.regra)
        if texto:
            saida.append((cab, texto))
    return saida


def verificar_numeracao(cabecalhos: list[str], regra: str) -> list[str]:
    """Problems found in the chapter numbering (empty list when 1..n is complete)."""
    if regra not in ("romano", "capitulo"):
        return []
    problemas = []
    for i, c in enumerate(cabecalhos):
        m = _RE_CAB_ROMANO.match(c) or _RE_CAB_CAPITULO.match(c)
        tok = m.group(1).upper() if m else ""
        n = 1 if tok in ("PRIMEIRO", "UNICO", "ÚNICO") else romano_para_int(tok)
        if n != i + 1:
            problemas.append(f"posicao {i + 1}: esperado {i + 1}, obtido {n}")
    return problemas


# ---------------------------------------------------------------------------
# Split of corpus A in contiguous blocks
# ---------------------------------------------------------------------------

MIN_CAPITULOS = 5
MIN_BYTES = 30 * 1024
MARGEM = 1


def _bloco_do_fim(tamanhos: list[int], fim: int, min_cap: int, min_bytes: int) -> int | None:
    """Largest start s so that tamanhos[s:fim] has >= min_cap units and >= min_bytes bytes."""
    soma = 0
    for s in range(fim - 1, -1, -1):
        soma += tamanhos[s]
        if fim - s >= min_cap and soma >= min_bytes:
            return s
    return None


def dividir_blocos(
    tamanhos: list[int],
    min_capitulos: int = MIN_CAPITULOS,
    min_bytes: int = MIN_BYTES,
    margem: int = MARGEM,
) -> list[str]:
    """Partition labels for the chapters of one work, in reading order.

    Layout: training, margin, validation block, margin, test block, with the test
    block at the end of the work. Each validation and test block is the shortest
    run of whole chapters with at least ``min_capitulos`` chapters and
    ``min_bytes`` bytes, and the training part must reach the same minimum. If
    both blocks do not fit, the work leaves the test set and keeps only
    validation; if that does not fit either, the whole work goes to training.
    """
    n = len(tamanhos)

    def cabe(fim_treino: int) -> bool:
        return fim_treino >= min_capitulos and sum(tamanhos[:fim_treino]) >= min_bytes

    s_teste = _bloco_do_fim(tamanhos, n, min_capitulos, min_bytes)
    if s_teste is not None and s_teste - margem > 0:
        fim_val = s_teste - margem
        s_val = _bloco_do_fim(tamanhos, fim_val, min_capitulos, min_bytes)
        if s_val is not None and cabe(s_val - margem):
            return (
                ["treino"] * (s_val - margem) + ["margem"] * margem + ["validacao"] * (fim_val - s_val)
                + ["margem"] * margem + ["teste"] * (n - s_teste)
            )
    s_val = _bloco_do_fim(tamanhos, n, min_capitulos, min_bytes)
    if s_val is not None and cabe(s_val - margem):
        return ["treino"] * (s_val - margem) + ["margem"] * margem + ["validacao"] * (n - s_val)
    return ["treino"] * n


# ---------------------------------------------------------------------------
# Proper names (training partition only)
# ---------------------------------------------------------------------------

_RE_PALAVRA = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*")
_FIM_FRASE = set(".!?…:;-«\"“(")
_ABREVIATURAS = {
    "D", "Dr", "Drs", "Sr", "Sra", "Srs", "S", "Exa", "Exm", "Exmo", "Illm", "Illmo", "V", "Mme", "Mr",
    "Mrs", "Snr", "Snra", "Fr", "St", "N", "Mlle", "Sras", "Dra",
}
_MESES_DIAS = set(_MESES.split("|")) | {
    "domingo", "segunda", "terça", "terca", "quarta", "quinta", "sexta", "sabbado", "sábado",
}


def _e_capitalizada(p: str) -> bool:
    return len(p) >= 2 and p[0].isupper() and not p.isupper()


def extrair_nomes(textos: list[str], min_ocorrencias: int = 2, razao_minuscula: float = 0.25) -> list[str]:
    """Capitalised words outside sentence start, extracted from training texts.

    A word is kept when it occurs capitalised outside sentence start at least
    ``min_ocorrencias`` times and its lowercase form occurs at most
    ``razao_minuscula`` times as often. Months, weekdays and abbreviations
    (``Dr``, ``Sr``) are dropped.
    A word right after an abbreviation such as ``D.`` or ``Sr.`` is not taken
    as sentence start.
    """
    cap: Counter[str] = Counter()
    minus: Counter[str] = Counter()
    for t in textos:
        for par in t.split("\n"):
            anterior_fim = 0
            ultima = ""
            inicio_frase = True
            for m in _RE_PALAVRA.finditer(par):
                p = m.group(0)
                entre = par[anterior_fim:m.start()]
                if any(c in _FIM_FRASE for c in entre):
                    abrev = entre.strip() == "." and ultima in _ABREVIATURAS
                    inicio_frase = not abrev
                if p.islower():
                    minus[p] += 1
                elif _e_capitalizada(p) and not inicio_frase:
                    cap[p] += 1
                inicio_frase = False
                anterior_fim = m.end()
                ultima = p
    nomes = [
        p for p, c in cap.items()
        if c >= min_ocorrencias and minus[p.lower()] <= razao_minuscula * c
        and p.lower() not in _MESES_DIAS and p not in _ABREVIATURAS
    ]
    return sorted(nomes, key=lambda s: (s.casefold(), s))


# ---------------------------------------------------------------------------
# Corpus B: classification of law summaries (ementas)
# ---------------------------------------------------------------------------

CLASSES_B = ("credito", "orcamento", "homenagem", "alteradora", "substantiva")
CLASSES_B_MANTIDAS = ("substantiva", "alteradora")

_RE_CREDITO = re.compile(
    r"\bcr[ée]ditos?\s+(especia(l|is)|suplementar(es)?|extraordin[áa]rios?)\b|^Abre\s+ao", re.I
)
_RE_ORCAMENTO = re.compile(
    r"\bLei\s+Or[çc]ament[áa]ria\b|diretrizes\s+para\s+a\s+elabora[çc][ãa]o|\bPlano\s+Plurianual\b"
    r"|^Estima\s+a\s+receita",
    re.I,
)
# Commemorative acts: names, honorary titles, patrons, commemorative dates and awareness months.
_RE_HOMENAGEM = re.compile(
    r"^(Inscreve|Denomina|Reconhece|Oficializa|D[áa]\s+o\s+nome)\b"
    r"|^(Confere|Concede|Outorga)\b.{0,80}\bt[íi]tulo\b"
    r"|^Declara\b.{0,120}\b(Patron[oa]|Capital\s+Nacional|Her[óo]i|Her[óo]ina)\b"
    r"|^Institui\b.{0,40}\b(Dia|Semana|M[êe]s|Ano|Bi[êe]nio|Data|Campanha|Pr[êe]mio|Janeiro|Fevereiro|Março"
    r"|Abril|Maio|Junho|Julho|Agosto|Setembro|Outubro|Novembro|Dezembro)\b",
    re.I,
)
_RE_ALTERADORA = re.compile(
    r"^(Altera|Alteram|Acrescenta|Acrescentam|Acresce|Revoga|Revogam|(D[áa]|Confere)\s+nova\s+reda[çc][ãa]o|Modifica"
    r"|Inclui|Insere|Prorroga|Restabelece|Torna\s+sem\s+efeito|Suprime|Renumera)\b",
    re.I,
)


def classificar_ementa(ementa: str) -> str:
    """Exclusive rule, checked in order: credito, orcamento, homenagem, alteradora, substantiva."""
    e = " ".join(unicodedata.normalize("NFC", ementa).split())
    if _RE_CREDITO.search(e):
        return "credito"
    if _RE_ORCAMENTO.search(e):
        return "orcamento"
    if _RE_HOMENAGEM.search(e):
        return "homenagem"
    if _RE_ALTERADORA.search(e):
        return "alteradora"
    return "substantiva"


_RE_CITACAO = re.compile(
    r"\b(Lei\s+Complementar|Decreto-Lei|Decreto\s+Legislativo|Medida\s+Provis[óo]ria|Decreto|Lei)s?"
    r"\s+n(?:[ºo°]|\.[ºo°]?)?s?\.?\s*(\d[\d.]*)",
    re.I,
)
_PREFIXO_CITACAO = {
    "lei complementar": "lcp",
    "decreto-lei": "del",
    "decreto legislativo": "dlg",
    "medida provisoria": "mpv",
    "medida provisória": "mpv",
    "decreto": "dec",
    "lei": "lei",
}


def primeira_lei_citada(ementa: str) -> str | None:
    """Identifier of the first act cited in the summary, as 'lei-10931' or 'del-2848'."""
    m = _RE_CITACAO.search(" ".join(ementa.split()))
    if m is None:
        return None
    tipo = _PREFIXO_CITACAO[" ".join(m.group(1).lower().split())]
    numero = m.group(2).replace(".", "")
    return f"{tipo}-{int(numero)}"


class UniaoBusca:
    """Union-find with path halving; the representative of a set is its smallest key."""

    def __init__(self) -> None:
        self.pai: dict[str, str] = {}

    def achar(self, x: str) -> str:
        self.pai.setdefault(x, x)
        while self.pai[x] != x:
            self.pai[x] = self.pai[self.pai[x]]
            x = self.pai[x]
        return x

    def unir(self, a: str, b: str) -> str:
        ra, rb = self.achar(a), self.achar(b)
        if ra == rb:
            return ra
        r, f = (ra, rb) if ra < rb else (rb, ra)
        self.pai[f] = r
        return r

    def grupos(self) -> dict[str, list[str]]:
        saida: dict[str, list[str]] = {}
        for x in list(self.pai):
            saida.setdefault(self.achar(x), []).append(x)
        return {k: sorted(v) for k, v in saida.items()}


# ---------------------------------------------------------------------------
# MinHash of word 5-grams with digits normalised
# ---------------------------------------------------------------------------

_PRIMO = (1 << 61) - 1
_RE_TOKEN = re.compile(r"\w+")


def shingles(texto: str, n: int = 5) -> set[int]:
    """Set of hashes of word n-grams, lowercase, with every digit mapped to 0."""
    toks = _RE_TOKEN.findall(re.sub(r"\d", "0", texto.lower()))
    if 0 < len(toks) < n:
        toks = toks + [""] * (n - len(toks))
    saida = set()
    for i in range(len(toks) - n + 1):
        h = hashlib.blake2b(" ".join(toks[i:i + n]).encode("utf-8"), digest_size=8).digest()
        saida.add(struct.unpack("<Q", h)[0] % _PRIMO)
    return saida


def _permutacoes(k: int, semente: int) -> list[tuple[int, int]]:
    perm = []
    for i in range(k):
        d = hashlib.blake2b(f"{semente}:{i}".encode(), digest_size=16).digest()
        a, b = struct.unpack("<QQ", d)
        perm.append((a % (_PRIMO - 1) + 1, b % _PRIMO))
    return perm


def assinatura_minhash(conj: set[int], k: int = 128, semente: int = 0) -> tuple[int, ...]:
    if not conj:
        return tuple([_PRIMO] * k)
    return tuple(min((a * h + b) % _PRIMO for h in conj) for a, b in _permutacoes(k, semente))


def jaccard_estimado(s1: tuple[int, ...], s2: tuple[int, ...]) -> float:
    return sum(x == y for x, y in zip(s1, s2)) / len(s1)


def jaccard(c1: set[int], c2: set[int]) -> float:
    if not c1 and not c2:
        return 1.0
    return len(c1 & c2) / len(c1 | c2)


def pares_quase_copia(
    conjuntos: dict[str, set[int]],
    limiar: float = 0.8,
    k: int = 128,
    bandas: int = 32,
    semente: int = 0,
) -> list[tuple[str, str, float]]:
    """Pairs with Jaccard >= limiar: LSH over MinHash signatures, then exact check.

    With 32 bands of 4 rows a pair at Jaccard 0.8 becomes a candidate with
    probability 1 - (1 - 0.8**4)**32, above 0.9999.
    """
    linhas = k // bandas
    assin = {d: assinatura_minhash(c, k, semente) for d, c in conjuntos.items() if c}
    candidatos: set[tuple[str, str]] = set()
    for b in range(bandas):
        baldes: dict[tuple[int, ...], list[str]] = {}
        for d, s in assin.items():
            baldes.setdefault(s[b * linhas:(b + 1) * linhas], []).append(d)
        for ids in baldes.values():
            ids.sort()
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    candidatos.add((ids[i], ids[j]))
    saida = []
    for a, b in sorted(candidatos):
        j = jaccard(conjuntos[a], conjuntos[b])
        if j >= limiar:
            saida.append((a, b, j))
    return saida


# ---------------------------------------------------------------------------
# Planalto pages
# ---------------------------------------------------------------------------

_BLOCOS = {
    "p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "blockquote", "center",
    "ul", "ol",
}
_IGNORAR = {"script", "style", "head", "title"}
_RISCADO = {"strike", "s", "del"}

_ANOTACOES = (
    r"Reda[çc][ãa]o\s+dad|Inclu[íi]d[oa]s?\s+pel|Inclus[ãa]o\s+dada|Acrescid[oa]s?\s+pel|Acrescentad[oa]s?\s+pel"
    r"|Vide\b|Vig\s?[êe]ncia|Revogad[oa]s?\b|Renumerad[oa]s?\b|Regulamento|Regulamenta[çc][ãa]o"
    r"|Produ[çc][ãa]o\s+de\s+efeitos?|Promulga[çc][ãa]o|Convers[ãa]o\s+da|Mensagem\s+de\s+veto|Texto\s+compilado"
    r"|Retificad[oa]|Suprimid[oa]|Transformad[oa]|Revigorad[oa]|Execu[çc][ãa]o\s+suspensa|Express[ãa]o\s+suprimida"
    r"|Derrubada\s+de\s+veto|Vetad[oa]\s+pel|Partes?\s+vetadas?|Prorroga[çc][ãa]o\s+de\s+vig|ADI\b|ADIN\b|ADC\b"
    r"|ADPF\b|Medida\s+Provis[óo]ria\s+n"
)
_RE_ANOTACAO = re.compile(rf"\s*\((?:{_ANOTACOES})[^()]*(?:\([^()]*\)[^()]*)*\)", re.I)
_RE_LINHA_EDITORIAL = re.compile(rf"^\(?(?:{_ANOTACOES})[^;:]{{0,80}}$", re.I)
# Link labels that the page inserts inside the text, without parentheses.
_RE_ROTULO = re.compile(
    r"\s*\(?Promulga[çc][ãa]o\s+(?:das\s+)?partes\s+vetadas\)?"
    r"|\s*\bVide\s+(?:Lei|Decreto|Medida\s+Provis[óo]ria|ADI|ADC|ADPF)[^;:()]{0,40}?\d[\d.\-/]*(?:,\s*de\s+\d{4})?",
    re.I,
)
# Navigation lines of the page layout.
_NAVEGACAO = {"não remover", "down", "up", "texto para impressão", "texto atualizado", "texto compilado", "*",
              "mensagem de veto", "vigência", "texto atualizado texto compilado", "regulamento"}
_RE_TITULO_LEI = re.compile(r"^LEI\s+N[º°o]\.?\s*[\d.]+", re.I)
# Title of each act type on its compiled page: the page header before it is dropped.
TITULOS = {"LEI": _RE_TITULO_LEI,
           "LCP": re.compile(r"^LEI\s+COMPLEMENTAR\s+N[º°o]\.?\s*[\d.]+", re.I),
           "DEC": re.compile(r"^DECRETO\s+N[º°o]\.?\s*[\d.]+", re.I)}
_RE_RODAPE = re.compile(r"^(\*\s*)?Este\s+texto\s+n[ãa]o\s+substitui", re.I)
_RE_PONTOS = re.compile(r"\.{4,}")


_VAZIAS = {"br", "img", "meta", "hr", "input", "link", "col", "area", "base", "wbr"}


class _ConversorHTML(HTMLParser):
    """HTML to text with an element stack, so that struck-through text is dropped
    whether it is marked by <strike>, <s>, <del> or by a line-through style."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.partes: list[tuple[str, bool]] = []  # (text, inside blockquote)
        self.pilha: list[tuple[str, bool]] = []  # (tag, struck)

    def _contar(self, pred) -> int:
        return sum(1 for t, r in self.pilha if pred(t, r))

    def _fechar(self, tag: str) -> None:
        if any(t == tag for t, _ in self.pilha):
            while self.pilha:
                t, _ = self.pilha.pop()
                if t == tag:
                    break

    def handle_starttag(self, tag, attrs):
        if tag in _BLOCOS:
            self.partes.append(("\n", False))
        if tag in _VAZIAS:
            return
        if tag == "p":
            self._fechar("p")  # an open <p> is closed implicitly by the next one
        estilo = " ".join((v or "") for k, v in attrs if k == "style").lower().replace(" ", "")
        self.pilha.append((tag, tag in _RISCADO or "line-through" in estilo))

    def handle_startendtag(self, tag, attrs):
        if tag in _BLOCOS:
            self.partes.append(("\n", False))

    def handle_endtag(self, tag):
        self._fechar(tag)
        if tag in _BLOCOS:
            self.partes.append(("\n", False))

    def handle_data(self, data):
        if self._contar(lambda t, r: t in _IGNORAR or r):
            return
        self.sup = self._contar(lambda t, r: t == "sup")
        self.citacao = self._contar(lambda t, r: t == "blockquote")
        if self.sup and data.strip() in ("o", "a"):
            data = {"o": "º", "a": "ª"}[data.strip()]
        # Source line breaks are layout only; paragraph breaks come from block tags.
        self.partes.append((data.replace("\r", " ").replace("\n", " "), self.citacao > 0))


def decodificar_planalto(bruto: bytes) -> str:
    """Planalto pages are ISO-8859-1 in practice; decode as cp1252, the superset that
    browsers apply to that label, fall back to latin-1, and accept valid UTF-8."""
    try:
        return bruto.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        return bruto.decode("cp1252")
    except UnicodeDecodeError:
        return bruto.decode("latin-1")


def limpar_planalto(bruto: bytes, titulo: re.Pattern = _RE_TITULO_LEI) -> tuple[str, int]:
    """Clean text of a compiled law page, and the bytes of it quoted in blockquotes.

    Struck-through text (revoked wording) is removed with its content, editorial
    annotations are removed, the page header before the law title is dropped and
    the text ends before the footer "Este texto nao substitui o publicado". `titulo` is the
    title of the act type, one of TITULOS.
    """
    conv = _ConversorHTML()
    conv.feed(decodificar_planalto(bruto))
    conv.close()
    linhas: list[tuple[str, bool]] = []
    atual: list[str] = []
    cit = False
    for txt, dentro in conv.partes + [("\n", False)]:
        for k, p in enumerate(txt.split("\n")):
            if k > 0:
                linhas.append(("".join(atual), cit))
                atual, cit = [], False
            if p:
                atual.append(p)
                cit = cit or (dentro and bool(p.strip()))
    saida: list[tuple[str, bool]] = []
    comecou = False
    for l, c in linhas:
        l = unicodedata.normalize("NFC", html.unescape(l).replace("\xa0", " "))
        l = " ".join(_RE_ROTULO.sub("", _RE_ANOTACAO.sub("", " ".join(l.split()))).split())
        if not l:
            continue
        if not comecou:
            if not titulo.match(l):
                continue
            comecou = True
        if _RE_RODAPE.match(l):
            break
        if _RE_LINHA_EDITORIAL.match(l) or l.lower() in _NAVEGACAO:
            continue
        saida.append((_RE_PONTOS.sub("...", l), c))
    texto = "\n\n".join(l for l, _ in saida)
    citado = sum(n_bytes(l) for l, c in saida if c)
    return texto, citado


def titulo_confere(texto: str, numero: int, titulo: re.Pattern = _RE_TITULO_LEI) -> bool:
    """True when the clean text starts with the title of act ``numero`` of the given type."""
    m = titulo.match(texto)
    return bool(m) and int(re.sub(r"\D", "", m.group(0))) == numero


# ---------------------------------------------------------------------------
# Split of corpus B by group hash
# ---------------------------------------------------------------------------


def balde_grupo(grupo: str) -> int:
    return int(hashlib.sha256(grupo.encode("utf-8")).hexdigest(), 16) % 100


def particao_por_hash(grupo: str) -> str:
    b = balde_grupo(grupo)
    if b < 90:
        return "treino"
    if b < 95:
        return "validacao"
    return "teste"

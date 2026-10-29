"""Items of the corpus B tasks (docs/decisao_avaliacao.md, section 3).

T2, provision inserted or own: the item is the caput of a complete article with the "Art. N" label
removed. Inserted when the article sits whole inside a quoted block, written into another norm;
own when it is an article of the law's own body that is not the object clause, the wrapper of an
amendment, the entry into force, the repeal clause or a vetoed article. The wrapper is left out
because its form gives the label away.

T3, adulterated text: the item is a complete article without its label, in the original and in an
adulterated version that differs by one operator, one item per applicable family. Four families: a term swapped within a closed
list of pairs with the same grammar, a number swapped in the "N (words)" pair with the words
rewritten in the same gender, a cross-reference to another article cited in the same law, and the
contents of two provisions of the same kind swapped with the labels kept in place. In the term and
number families the swaps preserve the frequency of each value among the adulterated texts, so
that a decider reading one text at a time cannot tell adulteration from word frequency.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Callable

NR = re.compile(r"^\s*\((NR|AC)\)", re.M)
ROTULO = re.compile(r"^Art\.\s*\d+[\dº°.]*(-[A-Z]+)?\.?\s*")
INICIO_ART = re.compile(r"(?m)^(?=Art\.\s*\d)")
RETICENCIAS = re.compile(r"(?m)^\s*\.{3,}|…")
EMBRULHO = re.compile(r"passa(m)? a vigorar|acrescid[oa]s?\b.*\bseguinte|seguinte reda[çc][ãa]o"
                      r"|^(Esta Lei|Revoga(m)?-se|Ficam? revogad)|entra(m)? em vigor")
MIN_PALAVRAS = 15
INSERIDO, PROPRIO = 1, 0

DISPOSITIVO = {"paragrafo": re.compile(r"^(§\s*\d+[º°]?\.?|Parágrafo único\.?)\s*"),
               "inciso": re.compile(r"^[IVXLC]+\s*[-\N{EN DASH}]\s*"),
               "alinea": re.compile(r"^[a-z]\)\s*")}
PENA = re.compile(r"^Pena\s*[-\N{EN DASH}]")
FINAL = re.compile(r"(\s*(;\s*(e|ou)|[;.:,]))?\s*$")
TERMOS = (("poderá", "deverá"), ("poderão", "deverão"), ("no mínimo", "no máximo"),
          ("Estados", "Municípios"), ("mensal", "anual"))
NUMERO = re.compile(r"(?<![\d.,])(\d{1,2})\s*\(([^()]+)\)")
REMISSAO = re.compile(r"\bart\.\s*(\d+)([º°])?(?![\d-])")
_UNI = ("", "um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito", "nove")
_DEZ = ("dez", "onze", "doze", "treze", "quatorze", "quinze", "dezesseis", "dezessete", "dezoito", "dezenove")
_DEZENAS = ("", "", "vinte", "trinta", "quarenta", "cinquenta", "sessenta", "setenta", "oitenta", "noventa")


def corpo_lei(texto: str) -> str:
    """Law text without the heading and the summary, the first two paragraphs of the document."""
    return "\n\n".join(texto.split("\n\n")[2:])


def _artigos(trecho: str) -> list[str]:
    return [a.strip() for a in INICIO_ART.split(trecho) if a.strip().startswith("Art")]


def _caput(artigo: str) -> str:
    return ROTULO.sub("", artigo.split("\n\n")[0], count=1).strip()


def _valido(c: str) -> bool:
    return len(c.split()) >= MIN_PALAVRAS and "VETADO" not in c and not RETICENCIAS.search(c)


def _h(*partes) -> str:
    return hashlib.sha256(":".join(map(str, partes)).encode()).hexdigest()


def citacoes(corpo: str) -> tuple[list[str], str] | None:
    """Top-level quoted blocks and the text outside them, by depth count, so that a quotation
    nested inside an inserted article does not close the block early. None when the quotes do not
    balance (for instance a closing mark used as the seconds sign of a coordinate): the label of
    every article of that law would be in doubt."""
    blocos, fora, prof, ini, ult = [], [], 0, 0, 0
    for i, ch in enumerate(corpo):
        if ch == "“":
            if prof == 0:
                fora.append(corpo[ult:i])
                ini = i + 1
            prof += 1
        elif ch == "”":
            prof -= 1
            if prof < 0:
                return None
            if prof == 0:
                blocos.append(corpo[ini:i])
                ult = i + 1
    if prof:
        return None
    fora.append(corpo[ult:])
    return blocos, NR.sub("", "\n\n".join(fora))


def _so_dispositivos(artigo: str) -> str:
    """The article up to the first paragraph that is not one of its provisions (a chapter heading
    or the rubric of the next article)."""
    blocos = artigo.split("\n\n")
    fim = next((i for i, b in enumerate(blocos[1:], 1)
                if not (PENA.match(b) or any(r.match(b) for r in DISPOSITIVO.values()))), len(blocos))
    return "\n\n".join(blocos[:fim]).strip()


def artigos_lei(doc: dict) -> list[tuple[str, int]] | None:
    """(article without its label, INSERIDO or PROPRIO) for the complete articles of a law, None
    when its quotes do not balance."""
    partes = citacoes(corpo_lei(doc["texto"]))
    if partes is None:
        return None
    blocos, fora = partes
    ins = [(ROTULO.sub("", a, count=1), INSERIDO) for b in blocos if not RETICENCIAS.search(b) for a in _artigos(b)]
    pro = [(ROTULO.sub("", a, count=1), PROPRIO) for a in _artigos(fora) if not EMBRULHO.search(_caput(a))]
    return ins + pro


def itens_t2(doc: dict) -> list[dict]:
    arts = artigos_lei(doc)
    if arts is None:
        return []
    base = {"doc_id": doc["doc_id"], "grupo": doc["grupo"], "particao": doc["particao"]}
    return ([{**base, "texto": c, "rotulo": r} for a, r in arts if r == INSERIDO for c in [_caput(a)] if _valido(c)]
            + [{**base, "texto": c, "rotulo": r} for a, r in arts if r == PROPRIO for c in [_caput(a)] if _valido(c)])


def extenso(n: int, feminino: bool = False) -> str:
    """Portuguese words of 1 to 99, in the gender of the counted noun."""
    u = lambda k: ("uma", "duas")[k - 1] if feminino and k in (1, 2) else _UNI[k]
    if n < 10:
        return u(n)
    if n < 20:
        return _DEZ[n - 10]
    return _DEZENAS[n // 10] + (f" e {u(n % 10)}" if n % 10 else "")


def _genero(n: int, palavras: str) -> str | None:
    """'m', 'f' or 'n' (the same in both genders) when the words spell n, None otherwise."""
    p = " ".join(palavras.lower().replace("catorze", "quatorze").split())
    m, f = extenso(n), extenso(n, True)
    return ("n" if m == f else "m") if p == m else "f" if p == f else None


def _trocar_termo(texto: str, a: str, b: str) -> str | None:
    if re.search(rf"(?<!\w){re.escape(b)}(?!\w)", texto):
        return None          # the target already occurs: the swap would repeat it
    m = re.search(rf"(?<!\w){re.escape(a)}(?!\w)(?!\s+Unidos)", texto)
    return None if m is None else texto[:m.start()] + b + texto[m.end():]


def _numero(texto: str):
    for m in NUMERO.finditer(texto):
        n = int(m.group(1))
        g = _genero(n, m.group(2)) if n > 0 else None
        if g:
            return m, n, g
    return None


def _remissao(texto: str, lei: str, chave: str) -> str | None:
    m = REMISSAO.search(texto)
    if m is None:
        return None
    outros = sorted({int(x.group(1)) for x in REMISSAO.finditer(lei)} - {int(m.group(1))})
    if not outros:
        return None
    k = min(outros, key=lambda v: _h(chave, v))
    rot = f"{k}{m.group(2) or 'º'}" if k <= 9 else str(k)
    return texto[:m.start(1)] + rot + texto[m.end():]


def _embaralhar(texto: str, chave: str) -> str | None:
    blocos = texto.split("\n\n")
    pares = []
    for tipo, r in DISPOSITIVO.items():
        idx = [i for i, b in enumerate(blocos[1:], 1) if r.match(b)]
        pares += [(tipo, i, j) for x, i in enumerate(idx) for j in idx[x + 1:]]
    if not pares:
        return None
    tipo, i, j = min(pares, key=lambda p: _h(chave, *p))
    r = DISPOSITIVO[tipo]

    def partes(b):
        rot = r.match(b).group(0)
        meio = b[len(rot):]
        fim = FINAL.search(meio).group(0)
        return rot, meio[:len(meio) - len(fim)], fim

    (ri, ci, fi), (rj, cj, fj) = partes(blocos[i]), partes(blocos[j])
    if ci == cj:
        return None
    blocos[i], blocos[j] = ri + cj + fi, rj + ci + fj
    return "\n\n".join(blocos)


def itens_t3(docs: list[dict], cabe: Callable[[str], bool] = lambda t: True, semente: int = 0) -> list[dict]:
    """Original and adulterated article pairs, one per family that applies to the article."""
    cand = defaultdict(list)
    for d in docs:
        for a in artigos_completos(d, cabe):
            chave = _h(semente, d["doc_id"], a)
            opcoes = {}
            termos = [(x, y) for p in TERMOS for x, y in (p, p[::-1]) if _trocar_termo(a, x, y)]
            if termos:
                opcoes["termo"] = min(termos, key=lambda t: _h(chave, *t))
            if _numero(a):
                opcoes["numero"] = True
            if (r := _remissao(a, d["texto"], chave)) is not None:
                opcoes["remissao"] = r
            if (e := _embaralhar(a, chave)) is not None:
                opcoes["embaralhamento"] = e
            for fam, op in opcoes.items():
                cand[fam].append({"doc_id": d["doc_id"], "grupo": d["grupo"], "particao": d["particao"],
                                  "original": a, "familia": fam, "_chave": _h(chave, fam), "_op": op})
    itens = [{**c, "adulterado": c["_op"]} for f in ("remissao", "embaralhamento") for c in cand[f]]
    # term family: as many swaps in each direction of a pair, so every term keeps its frequency
    por_par = defaultdict(lambda: defaultdict(list))
    for c in cand["termo"]:
        por_par[tuple(sorted(c["_op"]))][c["_op"]].append(c)
    for sentidos in por_par.values():
        k = min(len(v) for v in sentidos.values()) if len(sentidos) == 2 else 0
        for (x, y), v in sentidos.items():
            itens += [{**c, "adulterado": _trocar_termo(c["original"], x, y)} for c in sorted(v, key=lambda c: c["_chave"])[:k]]
    # number family: a cyclic shift of the values within each gender class, so the multiset of
    # numbers among the adulterated texts equals the one among the originals
    por_genero = defaultdict(list)
    for c in cand["numero"]:
        m, n, g = _numero(c["original"])
        por_genero[g].append((c, m, n))
    for g, v in por_genero.items():
        v.sort(key=lambda x: x[0]["_chave"])
        for x, (c, m, n) in enumerate(v):
            novo = v[(x + 1) % len(v)][2]
            if novo != n:
                rot = f"{novo} ({extenso(novo, g == 'f')})"
                itens.append({**c, "adulterado": c["original"][:m.start()] + rot + c["original"][m.end():]})
    itens = [{k: v for k, v in it.items() if not k.startswith("_")} for it in itens if cabe(it["adulterado"])]
    return sorted(itens, key=lambda it: _h(semente, it["doc_id"], it["original"], it["familia"]))


def artigos_completos(doc: dict, cabe: Callable[[str], bool] = lambda t: True) -> list[str]:
    """Complete articles of a law with a valid caput, cut at the first paragraph that is not one of
    their provisions."""
    return [a for a in (_so_dispositivos(x) for x, _ in artigos_lei(doc) or []) if _valido(_caput(a)) and cabe(a)]


def valores_numeros(docs: list[dict]) -> dict[str, set[int]]:
    """Distinct values of the "N (words)" pairs by gender class ('m', 'f', 'n'), the distractor pool of T1."""
    pool = defaultdict(set)
    for d in docs:
        for m in NUMERO.finditer(d["texto"]):
            n = int(m.group(1))
            if n > 0 and (g := _genero(n, m.group(2))):
                pool[g].add(n)
    return pool


def itens_t1(docs: list[dict], n_tokens: Callable[[str], int], pool: dict[str, set[int]],
             cabe: Callable[[str], bool] = lambda t: True, semente: int = 0, n_opcoes: int = 4) -> list[dict]:
    """Gaps over every number and cross-reference of each article. The distractors fill the gap
    with the article keeping its token count: numbers from the training pool in a compatible gender
    (only gender-neutral values for a gender-neutral gap), cross-references to other articles cited
    in the same law."""
    itens = []
    for d in docs:
        for a in artigos_completos(d, cabe):
            alvo = n_tokens(a)
            lacunas = []
            for m in NUMERO.finditer(a):
                n = int(m.group(1))
                if n > 0 and (g := _genero(n, m.group(2))):
                    vals = pool.get("n", set()) | (pool.get(g, set()) if g != "n" else set())
                    lacunas.append(("numero", m.start(), m.end(), a[m.start():m.end()],
                                    [f"{v} ({extenso(v, g == 'f')})" for v in sorted(vals - {n})]))
            citados = {int(x.group(1)) for x in REMISSAO.finditer(d["texto"])}
            for m in REMISSAO.finditer(a):
                n = int(m.group(1))
                lacunas.append(("remissao", m.start(1), m.end(), a[m.start(1):m.end()],
                                [f"{v}{m.group(2) or 'º'}" if v <= 9 else str(v) for v in sorted(citados - {n})]))
            for fam, ini, fim, certa, outras in lacunas:
                esq, dir_ = a[:ini], a[fim:]
                ok = [o for o in outras if o != certa and n_tokens(esq + o + dir_) == alvo]
                chave = _h(semente, d["doc_id"], a, ini)
                if len(ok) < n_opcoes - 1:
                    continue
                opcoes = sorted([certa] + sorted(ok, key=lambda o: _h(chave, o))[:n_opcoes - 1], key=lambda o: _h(chave, "ordem", o))
                itens.append({"doc_id": d["doc_id"], "grupo": d["grupo"], "particao": d["particao"], "familia": fam,
                              "artigo": a, "esquerda": esq, "direita": dir_, "opcoes": opcoes, "correta": opcoes.index(certa)})
    return itens


def itens_t4(docs: list[dict], cabe: Callable[[str], bool] = lambda t: True) -> list[dict]:
    """Continuation: the caput of each complete article split at half of its words, the prefix to
    condition on and the real continuation to score."""
    itens = []
    for d in docs:
        for a in artigos_completos(d, cabe):
            p = _caput(a).split(" ")
            k = len(p) // 2
            itens.append({"doc_id": d["doc_id"], "grupo": d["grupo"], "particao": d["particao"],
                          "prefixo": " ".join(p[:k]), "continuacao": " " + " ".join(p[k:])})
    return itens


def ngramas(ids: list[int], k: int) -> set[tuple[int, ...]]:
    return {tuple(ids[i:i + k]) for i in range(len(ids) - k + 1)}


def sem_sobreposicao(itens_ids: list[list[int]], treino_ids: list[list[int]], k: int = 13) -> list[bool]:
    """True for the items that share no k-token span with the training documents."""
    vistos = set().union(*(ngramas(d, k) for d in treino_ids))
    return [not (ngramas(ids, k) & vistos) for ids in itens_ids]


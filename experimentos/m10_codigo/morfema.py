"""Hand-designed morpheme rule for Portuguese, including the period spelling of the corpus.

A pre-token (GPT-2 pattern, possibly with a leading space) is split into
    [prefix] stem [derivational suffix] [verbal ending | nominal inflection]
by fixed lists, longest match first, matching on lowercase and cutting the original string, so the
pieces always concatenate back to the pre-token. Each layer cuts only when the remaining stem keeps a
minimum length. Variants, for measurement:
    "flexao"     verbal endings and nominal inflection only
    "derivacao"  plus one derivational suffix
    "prefixo"    plus one prefix from a short list of unambiguous prefixes
"""

DESINENCIAS_VERBAIS = [
    "ávamos", "íamos", "aríamos", "eríamos", "iríamos", "ássemos", "êssemos", "íssemos",
    "aremos", "eremos", "iremos", "aríeis", "assem", "essem", "issem", "ariam", "eriam", "iriam",
    "áramos", "êramos", "íramos", "avam", "aram", "eram", "iram", "arão", "erão", "irão",
    "ando", "endo", "indo", "asse", "esse", "isse", "aria", "eria", "iria",
    "ados", "adas", "idos", "idas", "ado", "ada", "ido", "ida",
    "ava", "avas", "ará", "erá", "irá", "amos", "emos", "imos", "aste", "este", "iste",
    "ou", "eu", "iu", "ar", "er", "ir",
]
FLEXAO_NOMINAL = ["es", "s", "a", "o"]
DERIVACIONAIS = [
    "mente", "cções", "cção", "ções", "ção", "sões", "são", "idades", "idade", "dades", "dade",
    "ismos", "ismo", "istas", "ista", "áveis", "ável", "íveis", "ível", "ância", "ência",
    "mentos", "mento", "agens", "agem", "eiros", "eiras", "eiro", "eira", "osos", "osas", "oso", "osa",
    "zinhos", "zinhas", "zinho", "zinha", "inhos", "inhas", "inho", "inha", "ezas", "eza",
    "uras", "ura", "ivos", "ivas", "ivo", "iva", "antes", "entes", "ante", "ente", "ores", "oras", "or", "ora",
    "ais", "al", "anças", "ança", "enças", "ença", "âncias", "ências",
]
# Words where a diminutive-looking ending is part of the lexical stem.
NAO_DIMINUTIVO = {
    "sobrinho", "sobrinha", "caminho", "vizinho", "vizinha", "carinho", "espinho", "moinho", "linho", "vinho",
    "ninho", "pinho", "padrinho", "madrinha", "rainha", "cozinha", "farinha", "galinha", "sardinha", "linha",
    "vinha", "mesquinho", "mesquinha", "adivinho", "adivinha", "focinho", "toucinho", "rebanho", "sozinho",
    "sozinha", "marinho", "marinha", "andorinha", "ladainha", "campainha", "bainha", "tainha",
}
PREFIXOS = ["contra", "entre", "sobre", "super", "inter", "anti", "auto", "des"]

for lista in (DESINENCIAS_VERBAIS, FLEXAO_NOMINAL, DERIVACIONAIS, PREFIXOS):
    lista.sort(key=len, reverse=True)

RADICAL_MIN = {"flexao": 4, "derivacao": 4, "prefixo": 4}
PALAVRA_MIN = 5


def _corta_sufixo(w: str, lista, minimo: int):
    low = w.lower()
    for s in lista:
        if low.endswith(s) and len(w) - len(s) >= minimo:
            return w[:-len(s)], w[-len(s):]
    return w, None


def segmentar(piece: str, variante: str = "derivacao") -> list[str]:
    lead = " " if piece.startswith(" ") else ""
    w = piece[len(lead):]
    if not w.isalpha() or len(w) < PALAVRA_MIN:
        return [piece]
    minimo = RADICAL_MIN[variante]
    fim = []
    lexical = w.lower().rstrip("s") in NAO_DIMINUTIVO
    if variante in ("derivacao", "prefixo") and not lexical:
        # derivational suffixes already carry their plural forms ("ções", "anças"), so they are
        # tried on the whole word before the inflection layer can split them
        w2, d = _corta_sufixo(w, DERIVACIONAIS, minimo)
        if d is not None:
            w, fim = w2, [d]
    if not fim:
        w, s = _corta_sufixo(w, DESINENCIAS_VERBAIS, minimo)
        if s is None:
            w, s = _corta_sufixo(w, FLEXAO_NOMINAL, minimo)
        if s is not None:
            fim.insert(0, s)
        if variante in ("derivacao", "prefixo") and not lexical:
            w, d = _corta_sufixo(w, DERIVACIONAIS, minimo)
            if d is not None:
                fim.insert(0, d)
    inicio = []
    if variante == "prefixo":
        low = w.lower()
        for p in PREFIXOS:
            if low.startswith(p) and len(w) - len(p) >= minimo:
                inicio, w = [w[:len(p)]], w[len(p):]
                break
    pedacos = [lead + inicio[0], w] if inicio else [lead + w]
    pedacos += fim
    assert "".join(pedacos) == piece, (piece, pedacos)
    return pedacos

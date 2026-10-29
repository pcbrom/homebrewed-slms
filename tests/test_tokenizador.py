"""Tokenizer contract U1 to U6 and the BPE-specific equivalence tests."""
import random
import unicodedata
from pathlib import Path

import pytest

from slm.tokenizador import (ESPECIAIS, N_ESP, BPEBiblioteca, BPEDidatico, TokenizadorBytes,
                             escolher_vocabulario, fracao_ausente)

RAIZ = Path(__file__).resolve().parents[1]
V = 600
# accented letters, a combining mark, a long dash, CJK and a 4-byte symbol built from code points
POOL = ("aãáâàçéêíóôõúü AEÇ\n\t.,;:!?'\"-1234567890 ñ[PAD][MASK]"
        + "".join(chr(c) for c in (0x2014, 0x6F22, 0x5B57, 0x1F642, 0x0301)))


def _amostra():
    textos = [unicodedata.normalize("NFC", p.read_text(encoding="utf-8")) for p in sorted((RAIZ / "docs").glob("*.md"))]
    return textos[:-2], textos[-2:]


@pytest.fixture(scope="module")
def tokenizadores():
    treino, _ = _amostra()
    return {"bytes": TokenizadorBytes(), "didatico": BPEDidatico().treinar(treino, V),
            "biblioteca": BPEBiblioteca().treinar(treino, V)}


def _aleatorio(n=200, seed=0):
    rng = random.Random(seed)
    return "".join(rng.choice(POOL) for _ in range(n))


@pytest.mark.parametrize("nome", ["bytes", "didatico", "biblioteca"])
def test_u1_ida_e_volta(tokenizadores, nome):
    tok = tokenizadores[nome]
    _, retido = _amostra()
    for texto in retido + [_aleatorio(seed=s) for s in range(20)]:
        assert tok.decode(tok.encode(texto)) == texto.encode("utf-8")


@pytest.mark.parametrize("nome", ["bytes", "didatico", "biblioteca"])
def test_u2_determinismo_e_persistencia(tokenizadores, nome, tmp_path):
    tok = tokenizadores[nome]
    texto = _amostra()[1][0]
    tok.salvar(tmp_path)
    outro = type(tok).carregar(tmp_path)
    assert tok.encode(texto) == tok.encode(texto) == outro.encode(texto)


@pytest.mark.parametrize("nome", ["bytes", "didatico", "biblioteca"])
def test_u3_faixa_de_ids(tokenizadores, nome):
    tok = tokenizadores[nome]
    ids = [i for t in _amostra()[1] for i in tok.encode(t)]
    assert all(0 <= i < tok.vocab_size for i in ids)


@pytest.mark.parametrize("nome", ["bytes", "didatico", "biblioteca"])
def test_u4_especiais_nunca_gerados(tokenizadores, nome):
    tok = tokenizadores[nome]
    texto = "inicio [PAD][BOS][EOS][MASK] fim " + _aleatorio(seed=99)
    assert not set(tok.encode(texto)) & set(ESPECIAIS.values())


@pytest.mark.parametrize("nome", ["bytes", "didatico", "biblioteca"])
def test_u5_decode_de_id_isolado(tokenizadores, nome):
    tok = tokenizadores[nome]
    for i in range(tok.vocab_size):
        assert isinstance(tok.decode([i]), bytes)


def test_u6_contrato_nao_pressupoe_bpe(tokenizadores):
    tok = tokenizadores["bytes"]
    assert tok.vocab_size == N_ESP + 256 and tok.especiais == ESPECIAIS


def test_bpe_fusoes_iguais(tokenizadores):
    assert tokenizadores["didatico"].merges == tokenizadores["biblioteca"].merges()


def test_bpe_ids_iguais_no_retido(tokenizadores):
    for texto in _amostra()[1]:
        assert tokenizadores["didatico"].encode(texto) == tokenizadores["biblioteca"].encode(texto)


def test_regra_de_vocabulario():
    treino, retido = _amostra()
    V_escolhido, medidas = escolher_vocabulario(treino, retido, grade=(300, 400, 600), limiar=1.0)
    assert V_escolhido == 600 and set(medidas) == {300, 400, 600}
    assert 0.0 <= fracao_ausente(BPEBiblioteca().treinar(treino, 400), retido) <= 1.0


def test_carregar_tokenizador_por_implementacao(tmp_path):
    from slm.corpus import RAIZ
    from slm.tokenizador import TokenizadorBytes, carregar_tokenizador, hash_tokenizador
    tok = carregar_tokenizador(RAIZ / "tokenizador" / "A")
    assert type(tok).__name__ == "BPEBiblioteca" and len(hash_tokenizador(tok)) == 64
    TokenizadorBytes().salvar(tmp_path / "bytes")
    tb = carregar_tokenizador(tmp_path / "bytes")
    assert tb.decode(tb.encode("ação")) == "ação".encode() and hash_tokenizador(tb) == ""


def test_carregar_tokenizador_externo(tmp_path):
    import json
    from slm.tokenizador import carregar_tokenizador, hash_tokenizador
    (tmp_path / "meu.py").write_text(
        "class Inverso:\n"
        "    vocab_size = 260\n"
        "    @classmethod\n"
        "    def carregar(cls, pasta):\n"
        "        return cls()\n"
        "    def encode(self, t):\n"
        "        return [4 + (255 - b) for b in t.encode()]\n"
        "    def decode(self, ids):\n"
        "        return bytes(255 - (i - 4) for i in ids if i >= 4)\n"
        "    def metadados(self):\n"
        "        return dict(implementacao='externo', sha256='x' * 64)\n", encoding="utf-8")
    (tmp_path / "metadados.json").write_text(json.dumps({"implementacao": "externo", "modulo": "meu.py", "classe": "Inverso"}))
    tok = carregar_tokenizador(tmp_path)
    assert tok.decode(tok.encode("coração")) == "coração".encode() and hash_tokenizador(tok) == "x" * 64


def test_conferir_contrato(tmp_path):
    from slm.tokenizador import conferir_contrato
    textos = _amostra()[1] + [_aleatorio(seed=1)]
    assert all(conferir_contrato(TokenizadorBytes(), textos, tmp_path).values())

    class Minusculas(TokenizadorBytes):
        def encode(self, texto):
            return super().encode(texto.lower())
    res = conferir_contrato(Minusculas(), textos)
    assert not res["U1 ida e volta exata"] and res["U3 ids dentro do vocabulário"]

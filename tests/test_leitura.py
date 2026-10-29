import csv
import random

import numpy as np
import pytest

from slm import leitura as lt


def test_palavras_letras_unicode_minusculas():
    assert lt.palavras("Olá, Mundo! O d'Ávila tem 3 cães_e gatos²x") == [
        "olá", "mundo", "o", "d", "ávila", "tem", "cães", "e", "gatos", "x"
    ]
    # decomposed input is normalized to NFC before splitting
    assert lt.palavras("José ação") == ["josé", "ação"]
    assert lt.palavras("123 -- !!") == []


def test_taxa_lexico_e_n1():
    lex = lt.lexico(["A casa amarela", "o gato"])
    assert lt.taxa_lexico("a casa azul do gato", lex) == pytest.approx(3 / 5)
    assert np.isnan(lt.taxa_lexico("123", lex))
    assert lt.nivel_n1([0.90, 0.92], [0.95, 0.95]).atingido
    r = lt.nivel_n1([0.80, 0.86], [0.95, 0.95])
    assert not r.atingido and r.diferenca == pytest.approx(-0.12)
    assert lt.nivel_n1([0.80], [0.95], limite=0.2).atingido


def _forca_bruta(treino_docs, consulta):
    grams = set()
    for d in treino_docs:
        ps = lt.palavras(d)
        for i in range(len(ps)):
            for j in range(i + 1, len(ps) + 1):
                grams.add(tuple(ps[i:j]))
    q = lt.palavras(consulta)
    return max((j - i for i in range(len(q)) for j in range(i + 1, len(q) + 1) if tuple(q[i:j]) in grams), default=0)


def test_maior_trecho_casos_montados():
    treino = ["o rato roeu a roupa do rei de roma", "era uma vez um gato"]
    idx = lt.IndiceTreino(treino)
    assert idx.maior_trecho("ontem o rato roeu a roupa azul") == 5
    assert idx.maior_trecho("nada aqui confere") == 0
    assert idx.maior_trecho("") == 0
    assert idx.maior_trecho("O RATO roeu a roupa do rei de roma") == 9
    # span across a document boundary does not count
    assert idx.maior_trecho("de roma era uma vez") == 3
    assert idx.maior_trecho("rei") == 1


def test_maior_trecho_contra_forca_bruta():
    rng = random.Random(0)
    vocab = [f"w{i}" for i in range(6)]
    treino = [" ".join(rng.choice(vocab) for _ in range(rng.randint(5, 40))) for _ in range(5)]
    idx = lt.IndiceTreino(treino)
    consultas = [" ".join(rng.choice(vocab + ["zz"]) for _ in range(rng.randint(0, 30))) for _ in range(60)]
    obtido = idx.maiores_trechos(consultas)
    esperado = [_forca_bruta(treino, q) for q in consultas]
    assert list(obtido) == esperado


def test_limiar_e_filtro_copia():
    k = lt.limiar_copia(np.arange(101))
    assert k == pytest.approx(99.0)
    f = lt.filtro_copia([1, 5, 12, 3], k=4)
    assert f.descartadas.tolist() == [False, True, True, False]
    assert f.taxa == 0.5 and f.intervalo[0] < 0.5 < f.intervalo[1]


def _lote(semente=11):
    itens = [
        {"texto": f"texto gerado numero {len(m)}{s}{j}", "modo": m, "ponto": 16, "semente": s, "amostra_id": f"{m}-{s}-{j}"}
        for m in ("difusao", "autoregressivo") for s in range(4) for j in range(3)
    ]
    pares = [(f"excerto real numero {j} aqui", lt.embaralhar(f"excerto real numero {j} aqui", j)) for j in range(20)]
    return itens, pares, *lt.montar_lote(itens, pares, semente)


def test_lote_cego_sem_vazamento(tmp_path):
    itens, pares, linhas, chave = _lote()
    p_lote, p_chave = lt.gravar_lote(tmp_path, 1, linhas, chave)
    with open(p_lote, encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        assert next(r) == ["item_id", "texto"]
        corpo = list(r)
    assert len(corpo) == len(itens) + 2 * len(pares) == 64
    assert all(len(l) == 2 for l in corpo)
    conteudo = p_lote.read_text(encoding="utf-8")
    for proibido in ("difusao", "autoregressivo", "modelo", "embaralhado", "ponto", "semente"):
        assert proibido not in conteudo
    # ids follow row order, so they say nothing beyond the random position
    assert [l[0] for l in corpo] == sorted(l[0] for l in corpo)
    chave_lida = lt.ler_chave(p_chave)
    assert set(chave_lida) == {l[0] for l in corpo}
    tipos = [chave_lida[l[0]]["tipo"] for l in corpo]
    assert tipos.count("modelo") == 24 and tipos.count("humano") == 20
    assert tipos != sorted(tipos)
    assert lt.itens_modelo(chave_lida, modo="difusao", semente=0).__len__() == 3


def test_lote_reprodutivel_e_ordem_depende_da_semente():
    a = _lote(1)[2]
    assert a == _lote(1)[2]
    assert [l["texto"] for l in a] != [l["texto"] for l in _lote(2)[2]]


def _gravar_respostas(caminho, notas):
    with open(caminho, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "N2", "N3", "N4"])
        for item_id, (a, b, c) in notas.items():
            w.writerow([item_id, a, b, c])


def test_respostas_filtro_nivel_e_concordancia(tmp_path):
    _, _, linhas, chave = _lote()
    rng = np.random.default_rng(0)
    modelo = lt.itens_modelo(chave)
    verdade = {i: int(v) for i, v in zip(modelo, rng.integers(0, 3, len(modelo)))}
    respostas = {}
    for av, erro_controle in (("autor", 0), ("aluno1", 1), ("aluno2", 5)):
        notas = {}
        for item_id, r in chave.items():
            if r["tipo"] == "modelo":
                v = verdade[item_id]
                notas[item_id] = (v, v, "" if av == "aluno1" and item_id == modelo[0] else v)
            elif r["tipo"] == "humano":
                notas[item_id] = (0, 0, 0) if r["par"] < erro_controle else (2, 2, 2)
            else:
                notas[item_id] = (0, 0, 0)
        _gravar_respostas(tmp_path / f"respostas_{av}_1.csv", notas)
        respostas[av] = lt.ler_respostas(tmp_path / f"respostas_{av}_1.csv")
    assert respostas["aluno1"][modelo[0]]["N4"] is None
    assert lt.filtro_avaliador(respostas["autor"], chave).aprovado
    assert lt.filtro_avaliador(respostas["aluno1"], chave).acertos == 19
    f = lt.filtro_avaliador(respostas["aluno2"], chave)
    assert f.acertos == 15 and not f.aprovado
    c = lt.concordancia(respostas, chave, "N4")
    assert c.alfa == pytest.approx(1.0) and c.aprovada
    assert lt.concordancia(respostas, chave, "N2", avaliadores=["aluno1", "aluno2"]).alfa == pytest.approx(1.0)
    m = lt.matriz_notas(respostas, modelo, "N4")
    assert m.shape == (3, 24) and np.isnan(m[1, 0])


def test_ler_respostas_rejeita_nota_invalida(tmp_path):
    _gravar_respostas(tmp_path / "r.csv", {"i001": (0, 3, 1)})
    with pytest.raises(ValueError):
        lt.ler_respostas(tmp_path / "r.csv")


def test_nivel_atingido():
    notas = np.array([[2] * 10 + [1, 0], [2] * 10 + [2, 0], [1] * 10 + [2, 0]], dtype=float)
    r = lt.nivel_atingido(notas)
    assert r.sucessos == 11 and r.atingido
    notas[:, :2] = 1
    r = lt.nivel_atingido(notas)
    assert r.sucessos == 9 and not r.atingido
    assert lt.nivel_atingido(np.full((3, 12), 2.0)).limite_inferior == pytest.approx(0.05 ** (1 / 12))
    assert lt.nivel_atingido(np.array([[2] * 10 + [0, 0]] * 3), minimo_itens=11).atingido is False
    assert lt.nivel_atingido([[2.0, np.nan], [2.0, np.nan]]).sucessos == 1

"""Unit tests for slm/dados.py (box 1)."""
import unicodedata

import pytest

from slm import dados as D

CAB = "The Project Gutenberg eBook of Teste\r\n\r\nLicense text here.\r\n\r\n"
START = "*** START OF THE PROJECT GUTENBERG EBOOK TESTE ***\r\n"
END = "*** END OF THE PROJECT GUTENBERG EBOOK TESTE ***\r\n"
RODAPE = "\r\nSection 1. General Terms of Use of Project Gutenberg.\r\n"


def _gutenberg(corpo: str) -> str:
    return CAB + START + corpo.replace("\n", "\r\n") + END + RODAPE


ROMANO = """
TITULO DA OBRA

POR

AUTOR




I

Do principio.

Era uma vez um homem que
morava no _Cattete_ e dizia:

--Bom dia, disse elle.



II.

O resto.

Outro paragrafo
em duas linhas.

     Verso primeiro,
     verso segundo.

FIM

INDICE

    I Do principio
    II O resto
"""


def test_corte_start_end_e_crlf():
    texto = D.cortar_gutenberg(_gutenberg("linha unica\n"))
    assert "Gutenberg" not in texto and "License" not in texto
    assert "\r" not in texto
    assert texto.strip() == "linha unica"


def test_corte_sem_marcas_falha():
    with pytest.raises(ValueError):
        D.cortar_gutenberg("sem marcas")


def test_nfc():
    decomposto = unicodedata.normalize("NFD", "\nCAPITULO I\n\nA mão é de João.\n")
    unidades = D.limpar_obra(_gutenberg(decomposto), D.Obra("1", "t", "T", "capitulo"))
    assert unicodedata.is_normalized("NFC", unidades[0][1])
    assert unidades[0][1] == "A mão é de João."


def test_corte_romano_folha_de_rosto_e_juncao():
    unidades = D.limpar_obra(_gutenberg(ROMANO), D.Obra("1", "t", "T", "romano"))
    assert [c for c, _ in unidades] == ["I", "II."]
    primeiro, segundo = unidades[0][1], unidades[1][1]
    assert "AUTOR" not in primeiro
    assert primeiro == "Do principio.\n\nEra uma vez um homem que morava no Cattete e dizia:\n\n--Bom dia, disse elle."
    assert segundo.startswith("O resto.\n\nOutro paragrafo em duas linhas.")
    assert "Verso primeiro,\nverso segundo." in segundo
    assert "INDICE" not in segundo and "FIM" not in segundo
    assert D.verificar_numeracao([c for c, _ in unidades], "romano") == []


def test_corte_capitulo_ignora_indice_inicial():
    corpo = "\nINDICE\n\nCAPITULO I\nCAPITULO II\n\nTITULO\n\nCAPITULO I\n\nTexto um.\n\nCAPITULO II\n\nTexto dois.\n"
    unidades = D.limpar_obra(_gutenberg(corpo), D.Obra("1", "t", "T", "capitulo"))
    assert [t for _, t in unidades] == ["Texto um.", "Texto dois."]


def test_corte_capitulo_primeiro_e_erro_de_grafia():
    corpo = "\nCAPITULO PRIMEIRO\n\nUm.\n\nCAPIULO II\n\nDois.\n"
    unidades = D.limpar_obra(_gutenberg(corpo), D.Obra("1", "t", "T", "capitulo"))
    assert [c for c, _ in unidades] == ["CAPITULO PRIMEIRO", "CAPIULO II"]
    assert D.verificar_numeracao([c for c, _ in unidades], "capitulo") == []


def test_numeracao_com_lacuna():
    assert D.verificar_numeracao(["I", "III"], "romano") == ["posicao 2: esperado 2, obtido 3"]


def test_corte_por_data_do_memorial():
    corpo = (
        "\nMEMORIAL\n\n1888\n\n9 de Janeiro.\n\nOra bem, faz hoje um anno.\n\n"
        "       *       *       *       *       *\n\nCinco horas da tarde.\n\n"
        "    9 de Janeiro.\n    «Mano,\n\n10 de Janeiro.\n\nFomos ao cemiterio.\n\n1889\n\n"
        "1 de Fevereiro, á noite.\n\nOutra entrada.\n\nSem data.\n\nUltima.\n"
    )
    unidades = D.limpar_obra(_gutenberg(corpo), D.Obra("1", "t", "T", "data"))
    assert [c for c, _ in unidades] == ["9 de Janeiro.", "10 de Janeiro.", "1 de Fevereiro, á noite.", "Sem data."]
    assert unidades[0][1].startswith("9 de Janeiro.\n\nOra bem")
    assert "Cinco horas da tarde." in unidades[0][1] and "*" not in unidades[0][1]
    assert "1889" not in unidades[1][1]


def test_corte_por_conto():
    corpo = (
        "\nADVERTENCIA\n\nPrefacio.\n\n\n\n\nPRIMEIRO CONTO[1]\n\n\nI\n\nTexto do primeiro.\n\n"
        "[1]Nota do autor.\n\n\nFIM DO PRIMEIRO CONTO\n\n\n\n\nSEGUNDO\n\nTexto do segundo.\n\nINDICE\n\n    PRIMEIRO\n"
    )
    unidades = D.limpar_obra(_gutenberg(corpo), D.Obra("1", "t", "T", "conto", r"^PRIMEIRO CONTO"))
    assert [t for _, t in unidades] == ["PRIMEIRO CONTO\n\nTexto do primeiro.\n\nNota do autor.", "SEGUNDO\n\nTexto do segundo."]


def test_italico_notas_e_hifens():
    t = D.limpar_texto_gutenberg(["_Dom_ veiu---e foi*", "[signature]", "frei ***."], "romano")
    assert t == "Dom veiu--e foi frei ***."


def test_divisao_layout_e_bloco_minimo():
    tam = [10_000] * 30
    rot = D.dividir_blocos(tam, min_capitulos=5, min_bytes=30_000, margem=1)
    assert rot[-5:] == ["teste"] * 5
    assert rot[-6] == "margem"
    assert rot[-11:-6] == ["validacao"] * 5
    assert rot[-12] == "margem"
    assert rot[:-12] == ["treino"] * 18


def test_divisao_bloco_cresce_ate_bytes():
    tam = [1_000] * 60 + [5_000] * 10
    rot = D.dividir_blocos(tam, min_capitulos=5, min_bytes=30_000, margem=1)
    assert rot.count("teste") == 6
    assert sum(b for b, r in zip(tam, rot) if r == "teste") >= 30_000
    assert sum(b for b, r in zip(tam, rot) if r == "validacao") >= 30_000


def test_divisao_obra_curta_sai_do_teste():
    rot = D.dividir_blocos([20_000] * 12, min_capitulos=5, min_bytes=30_000, margem=1)
    assert "teste" not in rot and rot.count("validacao") == 5 and rot.count("treino") == 6
    assert D.dividir_blocos([20_000] * 6) == ["treino"] * 6


def test_nomes_so_fora_do_inicio_de_frase():
    textos = [
        "Hoje encontrei o Escobar na rua. Escobar riu.",
        "Disse-me o Escobar que a Capitú viria. Depois saiu com o Sr. Bento.",
        "Vi a Capitú e o Bento. Janeiro chegou, e em Janeiro nada houve.",
        "Uma Rosa e outra rosa e mais uma rosa.",
    ]
    nomes = D.extrair_nomes(textos)
    assert "Escobar" in nomes and "Capitú" in nomes and "Bento" in nomes
    assert "Hoje" not in nomes and "Depois" not in nomes and "Janeiro" not in nomes
    assert "Rosa" not in nomes and "Sr" not in nomes


@pytest.mark.parametrize("ementa,classe", [
    ("Abre ao Orçamento Fiscal da União, em favor do Ministério da Educação, crédito especial no valor de R$ 1,00.", "credito"),
    ("Abre crédito extraordinário em favor de Ministérios.", "credito"),
    ("Estima a receita e fixa a despesa da União para o exercício financeiro de 2021.", "orcamento"),
    ("Dispõe sobre as diretrizes para a elaboração e a execução da Lei Orçamentária de 2020.", "orcamento"),
    ("Altera a Lei nº 13.898, de 11 de novembro de 2019, que dispõe sobre as diretrizes para a elaboração da Lei Orçamentária de 2020.", "orcamento"),
    ("Inscreve o nome de Fulano no Livro dos Heróis e Heroínas da Pátria.", "homenagem"),
    ("Institui o Dia Nacional do Forró.", "homenagem"),
    ("Denomina Viaduto Fulano o viaduto localizado na BR-101.", "homenagem"),
    ("Confere ao Município de X o título de Capital Nacional do Café.", "homenagem"),
    ("Altera a Lei nº 9.394, de 20 de dezembro de 1996, para dispor sobre X.", "alteradora"),
    ("Acrescenta art. 5º-A à Lei nº 8.069, de 13 de julho de 1990.", "alteradora"),
    ("Revoga dispositivos da Lei nº 1.234, de 1950.", "alteradora"),
    ("Dá nova redação ao art. 3º do Decreto-Lei nº 2.848, de 7 de dezembro de 1940.", "alteradora"),
    ("Dispõe sobre a proteção de dados pessoais e altera a Lei nº 12.965, de 23 de abril de 2014.", "substantiva"),
    ("Institui o Programa Nacional de Apoio às Microempresas.", "substantiva"),
    ("Institui o marco legal da microgeração distribuída.", "substantiva"),
    ("Institui a campanha Maio Laranja, a ser realizada no mês de maio.", "homenagem"),
    ("Outorga o título de Patrono da Aeronáutica a Fulano.", "homenagem"),
    ("Confere nova redação ao art. 1.520 da Lei nº 10.406, de 10 de janeiro de 2002.", "alteradora"),
    ("Altera a Lei nº 12.458, de 2011, para vedar a outorga do título de patrono a pessoas vivas.", "alteradora"),
])
def test_classificacao_de_ementas(ementa, classe):
    assert D.classificar_ementa(ementa) == classe


def test_primeira_lei_citada():
    assert D.primeira_lei_citada("Altera a Lei nº 9.394, de 20 de dezembro de 1996, e a Lei nº 8.069") == "lei-9394"
    assert D.primeira_lei_citada("Altera o Decreto-Lei nº 2.848, de 7 de dezembro de 1940") == "del-2848"
    assert D.primeira_lei_citada("Altera as Leis nºs 8.112, de 1990, e 9.784, de 1999") == "lei-8112"
    assert D.primeira_lei_citada("Altera a Lei Complementar nº 101, de 4 de maio de 2000") == "lcp-101"
    assert D.primeira_lei_citada("Institui o Programa X.") is None


def test_uniao_busca():
    uf = D.UniaoBusca()
    uf.unir("lei-2", "lei-9394")
    uf.unir("lei-3", "lei-9394")
    uf.unir("lei-5", "lei-6")
    uf.achar("lei-7")
    g = uf.grupos()
    assert g == {"lei-2": ["lei-2", "lei-3", "lei-9394"], "lei-5": ["lei-5", "lei-6"], "lei-7": ["lei-7"]}
    assert uf.achar("lei-9394") == "lei-2"


def _texto_lei(n: int, variante: str) -> str:
    base = " ".join(f"Art. {i}º O servidor da carreira {variante} recebe o adicional previsto no inciso {i} deste artigo." for i in range(1, 60))
    return f"LEI Nº {n}. {base}"


def test_minhash_quase_copia_detectada_e_diferente_nao():
    a = _texto_lei(13_001, "federal")
    b = _texto_lei(13_002, "federal").replace("Art. 30º", "Art. 30º Fica vedado.")
    c = " ".join(f"palavra{i} outra{i % 7} coisa{i % 11} termo{i % 13}" for i in range(400))
    conj = {"a": D.shingles(a), "b": D.shingles(b), "c": D.shingles(c)}
    assert D.jaccard(conj["a"], conj["b"]) >= 0.8
    pares = D.pares_quase_copia(conj, limiar=0.8)
    assert [(x, y) for x, y, _ in pares] == [("a", "b")]
    sa, sb, sc = (D.assinatura_minhash(conj[k]) for k in "abc")
    assert D.jaccard_estimado(sa, sb) >= 0.7
    assert D.jaccard_estimado(sa, sc) < 0.1


def test_shingles_normaliza_digitos():
    assert D.shingles("Lei 13.001 de 2019 altera tudo") == D.shingles("lei 14.999 de 2021 altera tudo")


def test_limpeza_planalto():
    html_doc = (
        "<html><head><title>L1</title><style>p{}</style></head><body>"
        "<p>Presidência da República</p><p>Texto compilado</p>"
        "<p><a>LEI Nº 13.970, DE\n 26 DE DEZEMBRO DE 2019</a></p>"
        "<p>Altera a Lei nº 10.931.</p>"
        "<p>Art. 1<sup>o</sup> Texto vigente. (Redação dada pela Lei nº 14.000, de 2020)</p>"
        "<p><strike>Art. 2º Texto revogado.</strike></p>"
        "<p>Art. 2º Novo texto. (Incluído pela Lei nº 14.001, de 2020) (Vide ADI 1)</p>"
        "<blockquote><p>“Art. 4º ......................” (NR)</p></blockquote>"
        "<p>Vigência</p>"
        "<p>Brasília, 27 de dezembro de 2019.</p>"
        "<p>Este texto não substitui o publicado no DOU de 27.12.2019</p><p>rodapé</p>"
        "</body></html>"
    ).encode("cp1252")
    texto, citado = D.limpar_planalto(html_doc)
    assert texto.split("\n\n") == [
        "LEI Nº 13.970, DE 26 DE DEZEMBRO DE 2019",
        "Altera a Lei nº 10.931.",
        "Art. 1º Texto vigente.",
        "Art. 2º Novo texto.",
        "“Art. 4º ...” (NR)",
        "Brasília, 27 de dezembro de 2019.",
    ]
    assert citado == D.n_bytes("“Art. 4º ...” (NR)")
    assert D.titulo_confere(texto, 13970) and not D.titulo_confere(texto, 13971)


def test_particao_por_hash_estavel():
    assert D.particao_por_hash("lei-9394") == D.particao_por_hash("lei-9394")
    contagem = {"treino": 0, "validacao": 0, "teste": 0}
    for i in range(5000):
        contagem[D.particao_por_hash(f"lei-{i}")] += 1
    assert 0.87 < contagem["treino"] / 5000 < 0.93


def test_planalto_riscado_por_estilo_e_rotulo_de_veto():
    html_doc = (
        "<body><p>LEI Nº 1, DE 2 DE JANEIRO DE 2020</p>"
        "<p><span style=\"color: black; text-decoration:line-through\">IV - texto antigo;</span></p>"
        "<p>IV - texto novo; <a>Promulgação partes vetadas</a>"
        "<p>V - outro item. (Vig ência encerrada)</p></body>"
    ).encode("cp1252")
    texto, _ = D.limpar_planalto(html_doc)
    assert texto.split("\n\n") == ["LEI Nº 1, DE 2 DE JANEIRO DE 2020", "IV - texto novo;", "V - outro item."]

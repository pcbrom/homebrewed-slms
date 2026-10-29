import re

from slm.tarefas import INSERIDO, PROPRIO, itens_t2, sem_sobreposicao

LONGO = "a autoridade competente publicará anualmente o relatório consolidado das ações executadas no período anterior"

TEXTO = f"""LEI Nº 1, DE 1 DE JANEIRO DE 2020

Altera a Lei nº 9, de 1 de janeiro de 1990, e dá outras providências.

O PRESIDENTE DA REPÚBLICA Faço saber que o Congresso Nacional decreta e eu sanciono a seguinte Lei:

Art. 1º Esta Lei altera a Lei nº 9, de 1 de janeiro de 1990, para tratar de {LONGO}.

Art. 2º A Lei nº 9, de 1 de janeiro de 1990, passa a vigorar acrescida do seguinte art. 5º-A:

“Art. 5º-A. O órgão gestor manterá cadastro nacional, com {LONGO}.

Parágrafo único. O cadastro será público.”

Art. 3º O art. 7º da Lei nº 9, de 1 de janeiro de 1990, passa a vigorar com a seguinte redação:

“Art. 7º ..........................................................

§ 2º O prazo será de trinta dias, contado da notificação, para {LONGO}.” (NR)

Art. 4º Fica instituído o programa de apoio, segundo o qual {LONGO}.

Art. 5º (VETADO).

Art. 6º Esta Lei entra em vigor na data de sua publicação.
"""

DOC = {"doc_id": "d1", "grupo": "g1", "particao": "teste", "texto": TEXTO}


def test_t2_separa_inserido_e_proprio_e_tira_rotulo_e_embrulho():
    itens = itens_t2(DOC)
    assert [(i["rotulo"], i["texto"][:22]) for i in itens] == [
        (INSERIDO, "O órgão gestor manterá"),
        (PROPRIO, "Fica instituído o prog"),
    ]
    assert all(not i["texto"].startswith("Art") and i["grupo"] == "g1" for i in itens)


def test_t2_descarta_citacao_parcial_com_reticencias():
    assert not any("prazo será de trinta dias" in i["texto"] for i in itens_t2(DOC))


def test_t2_citacao_aninhada_nao_fecha_o_bloco_e_aspas_soltas_descartam_a_lei():
    aninhada = TEXTO.replace("“Art. 5º-A. O órgão gestor manterá cadastro nacional,",
                             "“Art. 5º-A. O órgão gestor manterá o “Cadastro Nacional”,")
    aninhada = aninhada.replace("Parágrafo único. O cadastro será público.”",
                                f"Art. 5º-B. O cadastro será público e conterá {LONGO}.”")
    rot = [(i["rotulo"], i["texto"][:12]) for i in itens_t2({**DOC, "texto": aninhada})]
    assert rot == [(INSERIDO, "O órgão gest"), (INSERIDO, "O cadastro s"), (PROPRIO, "Fica institu")]
    solta = TEXTO.replace("segundo o qual", "no ponto 48°2’44,593”W, segundo o qual")
    assert itens_t2({**DOC, "texto": solta}) == []


def test_sobreposicao_de_k_tokens():
    treino = [list(range(100))]
    assert sem_sobreposicao([list(range(50, 63)), list(range(50, 62)) + [999]], treino, k=13) == [False, True]



def test_extenso_e_genero():
    from slm.tarefas import _genero, extenso
    assert [extenso(n) for n in (1, 2, 14, 21, 30, 99)] == ["um", "dois", "quatorze", "vinte e um", "trinta", "noventa e nove"]
    assert extenso(22, True) == "vinte e duas"
    assert (_genero(2, "duas"), _genero(2, "dois"), _genero(5, "cinco"), _genero(14, "catorze"), _genero(5, "seis")) == ("f", "m", "n", "n", None)


def _lei(i, corpo):
    return {"doc_id": f"d{i}", "grupo": f"g{i}", "particao": "treino",
            "texto": f"LEI Nº {i}\n\nDispõe sobre o tema.\n\n{corpo}"}


def test_t3_termo_equilibra_os_sentidos_e_nao_repete_o_alvo():
    from slm.tarefas import itens_t3
    poderia = [f"Art. 2º O órgão poderá editar normas complementares sobre {LONGO} {k}." for k in range(3)]
    deveria = [f"Art. 2º O órgão deverá editar normas complementares sobre {LONGO} {k}." for k in range(5)]
    ambos = ["Art. 2º O órgão poderá e deverá editar normas complementares sobre " + LONGO + "."]
    its = [i for i in itens_t3([_lei(k, t) for k, t in enumerate(poderia + deveria + ambos)]) if i["familia"] == "termo"]
    sent = [("poderá" in i["original"], "poderá" in i["adulterado"]) for i in its]
    assert sent.count((True, False)) == sent.count((False, True)) == 3
    assert all(i["original"].count(" ") == i["adulterado"].count(" ") for i in its)


def test_t3_numero_permuta_os_valores_e_refaz_o_extenso():
    from collections import Counter
    from slm.tarefas import itens_t3
    vals = [(2, "dois"), (1, "um"), (5, "cinco"), (30, "trinta"), (12, "doze"), (2, "duas")]
    docs = [_lei(k, f"Art. 3º O prazo de recurso será de {n} ({p}) dias, contados da intimação, para {LONGO}.")
            for k, (n, p) in enumerate(vals)]
    its = [i for i in itens_t3(docs) if i["familia"] == "numero"]
    # the feminine 2 is alone in its gender class and has no value to trade with
    assert len(its) == 5
    pega = lambda t: re.search(r"de (\d+) \((\w+)\) dias", t).groups()
    assert Counter(pega(i["original"]) for i in its) == Counter(pega(i["adulterado"]) for i in its)
    assert all(pega(i["original"]) != pega(i["adulterado"]) for i in its)


def test_t3_embaralhamento_mantem_rotulos_e_pontuacao():
    from slm.tarefas import _embaralhar
    art = f"Compete ao conselho, para {LONGO}:\n\nI - aprovar o plano anual;\n\nII - fiscalizar a execução; e\n\nIII - julgar os recursos."
    e = _embaralhar(art, "k")
    linhas = e.split("\n\n")
    assert [l[:4] for l in linhas[1:]] == ["I - ", "II -", "III "]
    assert [l[-1] for l in linhas[1:]] == [";", "e", "."] and e != art
    conteudo = lambda t: sorted(re.sub(r"^[IVX]+ - |(; e|;|\.)$", "", l) for l in t.split("\n\n")[1:])
    assert conteudo(e) == conteudo(art) and linhas[0] == art.split("\n\n")[0]


def test_t3_remissao_vem_da_mesma_lei():
    from slm.tarefas import _remissao
    lei = "Art. 4º Aplica-se o art. 7º. Art. 9º Ver art. 12."
    novo = _remissao("observado o disposto no art. 7º desta Lei", lei, "k")
    assert novo in {f"observado o disposto no art. {k} desta Lei" for k in ("4º", "9º", "12")}


def test_t1_lacunas_com_opcoes_de_mesmo_tamanho():
    from slm.tarefas import itens_t1, valores_numeros
    corpo = (f"Art. 3º O prazo de recurso será de 5 (cinco) dias, observado o art. 7º, para {LONGO}.\n\n"
             "Art. 7º Ver art. 4º. Art. 9º Ver art. 8º. Art. 11. Ver art. 6º.")
    doc = _lei(1, corpo)
    pool = valores_numeros([_lei(2, "prazos de 3 (três) dias, 6 (seis) dias, 8 (oito) dias e 2 (duas) horas")])
    assert pool == {"n": {3, 6, 8}, "f": {2}}
    n_tokens = lambda t: len(t.split())               # words stand in for tokens
    its = itens_t1([doc], n_tokens, pool)
    fams = {i["familia"]: i for i in its}
    assert set(fams) == {"numero", "remissao"}
    for i in its:
        certa = i["opcoes"][i["correta"]]
        assert i["esquerda"] + certa + i["direita"] == i["artigo"]
        assert len({n_tokens(i["esquerda"] + o + i["direita"]) for o in i["opcoes"]}) == 1 and len(set(i["opcoes"])) == 4
    assert sorted(fams["numero"]["opcoes"]) == sorted(["5 (cinco)", "3 (três)", "6 (seis)", "8 (oito)"])
    assert sorted(fams["remissao"]["opcoes"]) == ["4º", "6º", "7º", "8º"]


def test_t4_prefixo_e_continuacao_refazem_o_caput():
    from slm.tarefas import _caput, artigos_completos, itens_t4
    its = itens_t4([DOC])
    caputs = [_caput(a) for a in artigos_completos(DOC)]
    assert [i["prefixo"] + i["continuacao"] for i in its] == caputs
    assert all(abs(len(i["prefixo"].split()) - len(i["continuacao"].split())) <= 1 for i in its)


def test_t3_um_item_por_familia_aplicavel():
    from slm.tarefas import itens_t3
    corpo = (f"Art. 2º O órgão poderá cobrar taxa de 5 (cinco) reais, observado o art. 7º, para {LONGO}.\n\n"
             "Art. 7º Ver art. 4º e art. 9º para os casos omissos e para as hipóteses de recurso administrativo ao conselho nacional.")
    outros = [_lei(k, f"Art. 2º O órgão deverá cobrar taxa de {n} ({p}) reais para {LONGO}.")
              for k, (n, p) in enumerate([(3, "três"), (8, "oito")], start=10)]
    its = [i for i in itens_t3([_lei(1, corpo)] + outros) if i["doc_id"] == "d1" and i["original"].startswith("O órgão poderá")]
    assert {i["familia"] for i in its} == {"termo", "numero", "remissao"}

import math

import numpy as np
import pytest

from slm import estatistica as e

NAN = np.nan
# Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability", section C:
# 4 observers x 12 units, values 1 to 5, 7 missing.
KRIPP_2011 = [
    [1, 2, 3, 3, 2, 1, 4, 1, 2, NAN, NAN, NAN],
    [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, NAN, 3],
    [NAN, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, NAN],
    [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, NAN],
]


def _sintetico(rng, delta, n_grupos=10, docs_por_grupo=6, n_sementes=10, tau=0.01):
    """Per-document nats; groups and seed effects are drawn around a population Delta."""
    grupos = np.repeat(np.arange(n_grupos), docs_por_grupo)
    bytes_ = rng.integers(2_000, 20_000, size=grupos.size).astype(float)
    base = 1.5 + rng.normal(0, 0.1, n_grupos)[grupos]
    efeito_grupo = rng.normal(0, 0.01, n_grupos)[grupos]
    efeito_semente = rng.normal(0, tau, (n_sementes, 1))
    ruido = lambda: rng.normal(0, 0.02, (n_sementes, grupos.size))
    bpb_ar = base + ruido()
    bpb_dif = base + delta + efeito_grupo + efeito_semente + ruido()
    return bpb_dif * bytes_ * math.log(2), bpb_ar * bytes_ * math.log(2), bytes_, grupos


def test_bpb_razao_de_somas():
    docs = [
        {"doc_id": "a", "grupo": "g", "nats": math.log(2) * 10, "bytes": 10, "nats_sem_nomes": math.log(2) * 3, "bytes_sem_nomes": 6},
        {"doc_id": "b", "grupo": "g", "nats": math.log(2) * 30, "bytes": 10, "nats_sem_nomes": math.log(2) * 3, "bytes_sem_nomes": 4},
    ]
    assert e.bpb_documentos(docs) == pytest.approx(2.0)
    assert e.bpb_sem_nomes(docs) == pytest.approx(0.6)


def test_dados_pareados_alinha_por_doc_id():
    d1 = [{"doc_id": "a", "grupo": "x", "nats": 1.0, "bytes": 5}, {"doc_id": "b", "grupo": "y", "nats": 2.0, "bytes": 7}]
    d2 = list(reversed([{**d, "nats": d["nats"] + 1} for d in d1]))
    dp = e.dados_pareados([{"documentos": d1}], [d2])
    np.testing.assert_allclose(dp.nats_ar, [[2.0, 3.0]])
    with pytest.raises(ValueError):
        e.dados_pareados([d1], [[{**d1[0], "bytes": 6}, d1[1]]])


def test_bootstrap_cobre_valor_verdadeiro():
    rng = np.random.default_rng(1)
    delta, n_sim, cobre = -0.03, 200, 0
    for i in range(n_sim):
        nd, na, b, g = _sintetico(rng, delta)
        iv = e.bootstrap_delta(nd, na, b, g, R=2_000, semente=i)
        cobre += iv.inferior <= delta <= iv.superior
    assert cobre / n_sim >= 0.85


def test_bootstrap_estimativa_e_reprodutibilidade():
    rng = np.random.default_rng(2)
    nd, na, b, g = _sintetico(rng, 0.05)
    a = e.bootstrap_delta(nd, na, b, g, R=1_000, semente=7, guardar_replicas=True)
    c = e.bootstrap_delta(nd, na, b, g, R=1_000, semente=7)
    assert (a.inferior, a.superior) == (c.inferior, c.superior)
    esperado = np.mean(nd.sum(1) / (math.log(2) * b.sum()) - na.sum(1) / (math.log(2) * b.sum()))
    assert a.estimativa == pytest.approx(esperado)
    assert a.replicas.shape == (1_000,)
    assert a.inferior > 0
    assert e.ler_resultado(a.inferior, a.superior)[0] == "acima"


def test_cobertura_troca_sinal_em_dados_nulos():
    rng = np.random.default_rng(3)
    nd, na, b, g = _sintetico(rng, 0.0)
    cob = e.cobertura_troca_sinal(nd, na, b, g, n_nulos=200, R=1_000, semente=0)
    assert cob.contem_zero.shape == (200,)
    assert cob.cobertura >= 0.85


def test_teste_aleatorizacao():
    assert e.teste_aleatorizacao(np.full(10, 0.1)) == pytest.approx(2 / 2**10)
    assert e.teste_aleatorizacao([0.1, -0.1, 0.1, -0.1]) == pytest.approx(1.0)


def test_fator_sigma_sup():
    assert e.fator_sigma_sup(10) == pytest.approx(1.826, abs=5e-4)


@pytest.mark.parametrize("razao,n", [(0.5, 9), (0.75, 17), (1.0, 29)])
def test_numero_sementes_decisao(razao, n):
    r = e.numero_sementes(razao, 1.0, m=10, teto=30)
    assert r.resolvivel and r.n == n


def test_numero_sementes_nao_resolvivel_e_teto():
    r = e.numero_sementes(1.1, 1.0)
    assert not r.resolvivel and r.n is None
    assert e.numero_sementes(1.1, 1.0, teto=40).n is not None


def test_delta_min():
    m = e.delta_min([1.40, 1.42], [1.20, 1.18])
    assert m.valor == pytest.approx(0.055)
    assert m.positiva and m.erro_padrao > 0
    assert not e.delta_min(1.2, 1.3).positiva


def test_tost_identicos_e_deslocados():
    rng = np.random.default_rng(4)
    x = rng.normal(1.3, 0.01, 10)
    assert e.tost_pareado(x, x, margem=0.01).equivalente
    assert e.tost_pareado(x, x + rng.normal(0, 1e-4, 10), margem=0.01).equivalente
    r = e.tost_pareado(x, x + 0.05 + rng.normal(0, 1e-3, 10), margem=0.01)
    assert not r.equivalente and r.p > 0.05


def test_clopper_pearson():
    assert e.clopper_pearson_inferior(10, 12) == pytest.approx(0.562, abs=5e-4)
    assert e.clopper_pearson_inferior(0, 12) == 0.0
    lo, hi = e.clopper_pearson(10, 12)
    assert lo < 10 / 12 < hi
    assert e.clopper_pearson(0, 5)[0] == 0.0 and e.clopper_pearson(5, 5)[1] == 1.0
    lo, hi = e.clopper_pearson(3, 10)
    assert lo == pytest.approx(0.0667, abs=1e-4) and hi == pytest.approx(0.6525, abs=1e-4)


def test_krippendorff_exemplo_publicado():
    valores, o = e.matriz_coincidencias(KRIPP_2011)
    np.testing.assert_allclose(o.sum(1), [9, 13, 10, 5, 3])
    assert o[0, 0] == pytest.approx(7) and o[0, 1] == pytest.approx(4 / 3)
    assert e.alfa_krippendorff(KRIPP_2011, "nominal") == pytest.approx(0.743, abs=5e-4)
    assert e.alfa_krippendorff(KRIPP_2011, "ordinal") == pytest.approx(0.815, abs=5e-4)
    assert e.alfa_krippendorff(KRIPP_2011, "intervalar") == pytest.approx(0.849, abs=5e-4)


def test_krippendorff_casos_triviais():
    perfeito = [[0, 1, 2, 2, 0], [0, 1, 2, 2, 0], [0, 1, 2, NAN, 0]]
    assert e.alfa_krippendorff(perfeito) == pytest.approx(1.0)
    assert math.isnan(e.alfa_krippendorff([[2, 2, 2], [2, 2, 2]]))
    # ordinal alpha does not depend on the numeric labels, only on their order
    a = e.alfa_krippendorff(KRIPP_2011)
    b = e.alfa_krippendorff(np.asarray(KRIPP_2011) ** 3)
    assert a == pytest.approx(b)


def test_ler_resultado():
    assert e.ler_resultado(-0.2, -0.1) == ("abaixo", e.TEXTO_ABAIXO)
    assert e.ler_resultado(0.1, 0.2) == ("acima", e.TEXTO_ACIMA)
    assert e.ler_resultado(-0.1, 0.2) == ("contem_zero", "inconclusivo")

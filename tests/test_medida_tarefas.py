import numpy as np
import pytest

from slm import medida_tarefas as mt
from slm.medida_tarefas import PREFIXO_INSERCAO, agregar, contribuicoes, holm, ic_selvagem, limiar_balanceado, pontuar

# toy scorer: every token costs its own value in nats, [EOS] costs 1
codificar = lambda t: [ord(c) % 7 + 1 for c in t]
logp = lambda seqs: [-np.array([*s, 1.0], dtype=float) for s in seqs]
base = {"doc_id": "d", "grupo": "g", "particao": "externo"}


def test_t1_escolhe_a_opcao_de_maior_verossimilhanca():
    it = {**base, "familia": "numero", "esquerda": "a", "direita": "b", "opcoes": ["xx", "aa", "zz", "qq"], "correta": 1}
    custo = [sum(codificar("a" + o + "b")) for o in it["opcoes"]]
    r = pontuar("t1", [it], logp, codificar)[0]
    assert r["escolha"] == int(np.argmin(custo)) and r["acerto"] == (r["escolha"] == 1)


def test_t2_escore_e_diferenca_das_sequencias_inteiras():
    it = {**base, "texto": "abc", "rotulo": 1}
    r = pontuar("t2", [it], logp, codificar)[0]
    esperado = -(sum(codificar(PREFIXO_INSERCAO + "abc")) + 1) + (sum(codificar("Art. 1º abc")) + 1)
    assert r["escore"] == pytest.approx(esperado)


def test_t4_conta_so_a_continuacao_sem_eos():
    it = {**base, "prefixo": "ab", "continuacao": " cd"}
    r = pontuar("t4", [it], logp, codificar)[0]
    assert r["nats"] == pytest.approx(sum(codificar(" cd"))) and r["bytes"] == 3


def test_limiar_e_contribuicoes_somam_a_acuracia_balanceada():
    res = [{"grupo": g, "rotulo": y, "escore": e} for g, y, e in
           [("a", 1, 3.0), ("a", 0, -1.0), ("b", 1, 0.5), ("b", 0, 2.0), ("c", 0, -2.0), ("c", 0, -3.0)]]
    lim = limiar_balanceado([r["escore"] for r in res], [r["rotulo"] for r in res])
    c = contribuicoes("t2", res, lim)
    pred = [(r["escore"] > lim) == bool(r["rotulo"]) for r in res]
    ab = (np.mean([p for p, r in zip(pred, res) if r["rotulo"]]) + np.mean([p for p, r in zip(pred, res) if not r["rotulo"]])) / 2
    assert agregar("t2", list(c.values())) == pytest.approx(ab) and ab == pytest.approx(0.875)


def test_sinal_holm_e_intervalo():
    assert mt.teste_sinal([1, 2, 0, -1, 3]) == (3, 4, pytest.approx(0.625))
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    x = np.arange(20.0)
    lo, hi = ic_selvagem(x.mean(), mt.residuos_media(x), R=4000)
    assert lo < 9.5 < hi and hi - lo < 6
    # a constant classifier has balanced accuracy 0.5 in every sample: a point interval
    res = [{"grupo": g, "rotulo": y, "acerto": y == 0} for g, y in [("a", 1), ("a", 0), ("b", 0), ("b", 0), ("c", 1)]]
    e = mt.residuos_t2(res, ["a", "b", "c"])
    assert np.allclose(e, 0) and ic_selvagem(0.5, e, R=200) == pytest.approx((0.5, 0.5))
    res[0]["acerto"] = True
    assert not np.allclose(mt.residuos_t2(res, ["a", "b", "c"]), 0)


def test_resumo_e_comparacao_entre_pontuadores():
    rng = np.random.default_rng(0)
    res = lambda p: [{"grupo": f"g{k % 30}", "acerto": bool(rng.random() < p)} for k in range(600)]
    bom = mt.resumo_pontuador("t3", [res(0.8), res(0.8)])
    ruim = mt.resumo_pontuador("t3", [res(0.5)])
    c = mt.comparar("t3", bom, ruim, R=2000)
    assert 0.2 < c["diferenca"] < 0.4 and c["ic95"][0] > 0 and c["p_sinal"] < 0.01
    t4 = lambda b: [{"grupo": f"g{k % 10}", "nats": b * 0.693147 * 10, "bytes": 10} for k in range(50)]
    assert mt.comparar("t4", mt.resumo_pontuador("t4", [t4(1.0)]), mt.resumo_pontuador("t4", [t4(1.5)]))["diferenca"] == pytest.approx(0.5, rel=1e-5)


def test_icc1():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(40), rng.integers(3, 12, 40))
    efeito = rng.normal(0, 1, 40)[g]
    assert abs(mt.icc1(rng.normal(0, 1, g.size), g)) < 0.1
    assert mt.icc1(efeito + rng.normal(0, 0.5, g.size), g) > 0.7

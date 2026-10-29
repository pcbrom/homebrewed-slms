import pytest
import torch

from slm.mascaramento import mascarar
from slm.medida_rede import _comum, amostras, nats_resposta, pontuar_difusao
from slm.modelo import ConfigRede, Transformer
from slm.perda import _ce

L = 16


def _modelo():
    torch.manual_seed(0)
    return Transformer(ConfigRede(vocab_size=12, L=L)).eval()


def test_amostras_estratificadas_e_reprodutiveis():
    l, m = amostras(5, 10, 3)
    assert (m.sum(1) == l).all() and set(l.tolist()) == {1, 2, 3, 4, 5}
    l2, m2 = amostras(5, 10, 3)
    assert torch.equal(l, l2) and torch.equal(m, m2)


def test_resposta_de_um_token_e_a_condicional_exata():
    mod = _modelo()
    ids = [5, 6, 7, 8, 9]
    n = nats_resposta(mod, [(ids, 2, 3, 0)], L, K=4)[0]
    w = torch.full((1, L), 2)
    w[0, :5] = torch.tensor(ids)
    m = torch.zeros((1, L), dtype=torch.bool)
    m[0, 2] = True
    with torch.no_grad():
        exato = float(_ce(mod(mascarar(w, m), False), w)[0, 2])
    assert n == pytest.approx(exato, rel=1e-5)


def test_comum_e_mesmas_mascaras_entre_opcoes():
    assert _comum([1, 2, 3, 9, 5], [1, 2, 4, 4, 5]) == (2, 1)
    assert _comum([1, 2], [1, 2, 3]) == (2, 0)
    mod = _modelo()
    a, b = nats_resposta(mod, [([5, 6, 7, 8], 1, 3, 7), ([5, 6, 7, 8], 1, 3, 7)], L, K=8)
    assert a == b


def test_pontuar_difusao_t1_e_t4():
    mod = _modelo()
    cod = lambda t: [4 + ord(c) % 8 for c in t]
    base = {"doc_id": "d", "grupo": "g", "particao": "externo"}
    t1 = [{**base, "familia": "numero", "artigo": "abXYcd", "esquerda": "ab", "direita": "cd",
           "opcoes": ["XY", "ZW", "QR", "ST"], "correta": 0}]
    r = pontuar_difusao("t1", t1, mod, cod, L, K=8)[0]
    assert 0 <= r["escolha"] < 4
    t4 = [{**base, "prefixo": "abc", "continuacao": " de"}]
    r = pontuar_difusao("t4", t4, mod, cod, L, K=8)[0]
    assert r["nats"] > 0 and r["bytes"] == 3

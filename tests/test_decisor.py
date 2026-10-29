import numpy as np
import pytest
import torch

from slm.decisor import ajustar, ece, limiar_seletivo, logits, lote, probabilidades, temperatura
from slm.modelo import ConfigRede, Transformer

L = 16


def test_lote_preenche_com_eos_e_conta_o_eos():
    ids, n = lote([[5, 6], [7]], 4)
    assert ids.tolist() == [[5, 6, 2, 2], [7, 2, 2, 2]] and n.tolist() == [3, 2]


def test_estados_alimentam_o_forward():
    torch.manual_seed(0)
    m = Transformer(ConfigRede(vocab_size=12, L=L)).eval()
    x = torch.randint(4, 12, (2, L))
    with torch.no_grad():
        assert torch.allclose(m(x, False)[..., 4:], (m.estados(x, False) @ m.emb.weight.T)[..., 4:])


def test_decisor_aprende_regra_separavel_nos_dois_modos():
    torch.manual_seed(0)
    enc = Transformer(ConfigRede(vocab_size=12, L=L, dropout=0.0))
    rng = np.random.default_rng(0)
    seqs = [list(rng.integers(4, 8, 6)) if k % 2 else list(rng.integers(8, 12, 6)) for k in range(64)]
    y = np.array([k % 2 for k in range(64)])
    for causal in (True, False):
        perdas = []
        dec = ajustar(enc, causal, seqs, y, lr=3e-3, L=L, epocas=6, perdas=perdas)
        assert (logits(dec, seqs, L).argmax(1) == y).mean() > 0.9
        # one loss per step, 4 batches of 16 per epoch, falling over training
        assert len(perdas) == 24 and np.mean(perdas[-4:]) < np.mean(perdas[:4])


def test_calibracao():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    z = np.stack([np.zeros(4000), np.where(y == 1, 1.0, -1.0) + rng.normal(0, 1, 4000)], 1) * 4
    T = temperatura(z, y)
    assert T > 1.5 and ece(probabilidades(z, T), y) < ece(probabilidades(z), y)
    conf = np.array([0.99, 0.95, 0.9, 0.8, 0.7])
    assert limiar_seletivo(conf, np.array([1, 1, 1, 0, 1]), alvo=0.75) == 0.7
    assert limiar_seletivo(conf, np.array([1, 1, 1, 0, 0]), alvo=0.95) == 0.9
    assert limiar_seletivo(conf, np.array([0, 0, 0, 0, 0])) == float("inf")

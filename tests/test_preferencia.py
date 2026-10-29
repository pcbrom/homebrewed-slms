"""DPO in both modes on a tiny float64 network: zero margin at start, exact AR log-probability,
unbiased diffusion ELBO against enumeration, and a margin that grows with training."""
import copy
import itertools
import math

import pytest
import torch

from slm.modelo import ConfigRede, Transformer
from slm.perda import _ce, perda_ar
from slm.preferencia import elbo_difusao, logp_ar, mascaras_y, montar_janelas, perda_dpo
from slm.mascaramento import mascarar

V, L = 12, 6


@pytest.fixture(scope="module")
def rede():
    torch.manual_seed(0)
    return Transformer(ConfigRede(vocab_size=V, L=L, n_layer=1, d=16, n_head=2)).double().eval()


def _pares():
    x = [[4, 5], [6, 7, 8]]
    yg = [[9, 10, 11], [9, 9]]
    yp = [[11, 10, 9], [10, 4]]
    return montar_janelas(x, yg, L), montar_janelas(x, yp, L)


def test_montar_janelas_marca_so_a_continuacao():
    w, m = montar_janelas([[4, 5, 6, 7]], [[8, 9, 10]], 5)
    assert w[0].tolist() == [6, 7, 8, 9, 10]
    assert m[0].tolist() == [False, False, True, True, True]


def test_logp_ar_coincide_com_a_perda_do_pacote(rede):
    (w, _), _ = _pares()
    tudo = torch.ones_like(w, dtype=torch.bool)
    assert torch.allclose(logp_ar(rede, w, tudo), -perda_ar(rede, w)[1], atol=1e-12)


def test_mascaras_ficam_na_continuacao_e_estratificam(rede):
    (_, m), _ = _pares()
    g = torch.Generator().manual_seed(1)
    masc = mascaras_y(m, 8, g)
    assert bool((masc & ~m).sum() == 0)
    tam = masc.sum(dim=2)
    assert bool((tam >= 1).all()) and bool((tam <= m.sum(dim=1)).all())


@pytest.mark.parametrize("modo", ["autoregressivo", "difusao"])
def test_margem_zero_e_perda_log2_no_inicio(rede, modo):
    (wg, mg), (wp, mp) = _pares()
    perda, s, _ = perda_dpo(rede, copy.deepcopy(rede), wg, mg, wp, mp, modo, beta=0.1, n=4,
                            gerador=torch.Generator().manual_seed(2))
    assert torch.equal(s, torch.zeros_like(s))
    assert abs(float(perda.detach()) - math.log(2)) < 1e-12


def test_elbo_difusao_nao_viesada_contra_enumeracao(rede):
    (w, m), _ = _pares()
    w, m = w[:1], m[:1]
    pos = m[0].nonzero().squeeze(1).tolist()
    exato = 0.0
    with torch.no_grad():
        for l in range(1, len(pos) + 1):
            subs = list(itertools.combinations(pos, l))
            for S in subs:
                masc = torch.zeros_like(m)
                masc[0, list(S)] = True
                ce = float((_ce(rede(mascarar(w, masc), False), w) * masc).sum())
                exato += -(len(pos) / l) * ce / len(subs) / len(pos)
        amostras = torch.cat([elbo_difusao(rede, w, m, mascaras_y(m, 6, torch.Generator().manual_seed(s)))
                              for s in range(400)])
    media, ep = float(amostras.mean()), float(amostras.std() / math.sqrt(len(amostras)))
    assert abs(media - exato) < 4 * ep + 1e-9


@pytest.mark.parametrize("modo", ["autoregressivo", "difusao"])
def test_dpo_aumenta_a_margem(rede, modo):
    (wg, mg), (wp, mp) = _pares()
    pol, ref = copy.deepcopy(rede), copy.deepcopy(rede)
    opt = torch.optim.SGD(pol.parameters(), lr=0.5)
    g = torch.Generator().manual_seed(3)
    for _ in range(30):
        perda, s, _ = perda_dpo(pol, ref, wg, mg, wp, mp, modo, beta=0.5, n=4, gerador=g)
        opt.zero_grad()
        perda.backward()
        opt.step()
    with torch.no_grad():
        _, s, acc = perda_dpo(pol, ref, wg, mg, wp, mp, modo, beta=0.5, n=16, gerador=torch.Generator().manual_seed(9))
    assert acc == 1.0 and bool((s > 0).all())

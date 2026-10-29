from collections import Counter

import numpy as np
import pytest

from slm.ngrama import BOS, D_RESERVA, EOS, FORA, NGramaKN, descontos

V = 14


def _docs(semente=0, n=40):
    rng = np.random.default_rng(semente)
    # a skewed distribution so that counts of 1 to 4 all occur
    p = rng.dirichlet(np.full(V - 4, 0.4))
    return [list(rng.choice(np.arange(4, V), size=rng.integers(3, 30), p=p)) for _ in range(n)]


class KNDireto:
    """Second implementation, written from the formulas with Counters, no vectorisation."""

    def __init__(self, ordem, docs):
        self.n = ordem
        seqs = [[BOS] * (ordem - 1) + d + [EOS] for d in docs]
        topo = Counter(tuple(s[i - ordem + 1:i + 1]) for s in seqs for i in range(ordem - 1, len(s)))
        self.cont = {ordem: topo}
        for m in range(ordem - 1, 0, -1):
            self.cont[m] = Counter(g[1:] for g in self.cont[m + 1])
        self.d = {m: descontos(np.array(list(c.values()))) for m, c in self.cont.items()}
        self.previstos = [w for w in range(V) if w not in FORA]

    def prob(self, ctx, w, m=None):
        m = self.n if m is None else m
        baixo = 1 / len(self.previstos) if m == 1 else self.prob(ctx, w, m - 1)
        h = tuple(ctx[len(ctx) - m + 1:]) if m > 1 else ()
        filhos = {g: c for g, c in self.cont[m].items() if g[:-1] == h}
        total = sum(filhos.values())
        if total == 0:
            return baixo
        d = self.d[m]
        desc = lambda c: d[min(c, 3) - 1]
        massa = sum(desc(c) for c in filhos.values())
        c = filhos.get(h + (w,), 0)
        return max(c - desc(c), 0) / total + massa / total * baixo if c else massa / total * baixo


@pytest.mark.parametrize("ordem", [1, 2, 3, 4])
def test_distribuicao_soma_um_em_contextos_vistos_e_nao_vistos(ordem):
    m = NGramaKN(ordem, V).treinar(_docs())
    rng = np.random.default_rng(1)
    contextos = [tuple([BOS] * (ordem - 1))] + [tuple(rng.integers(4, V, size=ordem - 1)) for _ in range(20)]
    for ctx in contextos:
        p = m.distribuicao(ctx)
        assert p.sum() == pytest.approx(1.0, abs=1e-12)
        assert (p[list(FORA)] == 0).all() and (p[[w for w in range(V) if w not in FORA]] > 0).all()


@pytest.mark.parametrize("ordem", [1, 2, 3])
def test_igual_a_implementacao_direta(ordem):
    docs = _docs(semente=2, n=25)
    m, ref = NGramaKN(ordem, V).treinar(docs), KNDireto(ordem, docs)
    rng = np.random.default_rng(3)
    for _ in range(30):
        ctx = tuple(rng.integers(4, V, size=ordem - 1)) if rng.random() < 0.5 else tuple(docs[0][:ordem - 1])
        for w in [EOS] + list(range(4, V)):
            assert m.prob(ctx, w) == pytest.approx(ref.prob(ctx, w), rel=1e-12)


def test_nats_e_soma_das_condicionais_e_aprende_alem_do_uniforme():
    docs = _docs()
    m = NGramaKN(3, V).treinar(docs[:30])
    d = docs[0]
    seq = [BOS, BOS] + d + [EOS]
    esperado = -sum(np.log(m.prob(tuple(seq[i - 2:i]), seq[i])) for i in range(2, len(seq)))
    assert m.nats(d) == pytest.approx(esperado)
    uniforme = np.log(V - len(FORA))
    nats_ret = sum(m.nats(x) for x in docs[30:])
    tokens_ret = sum(len(x) + 1 for x in docs[30:])
    assert nats_ret / tokens_ret < uniforme


def test_descontos_de_reserva_quando_faltam_contagens():
    assert descontos(np.array([1, 1, 2])) == D_RESERVA
    d = descontos(np.array([1] * 50 + [2] * 20 + [3] * 10 + [4] * 5 + [9] * 3))
    assert all(0 < dk < k for dk, k in zip(d, (1, 2, 3)))


def test_geracao_termina_no_eos_e_so_produz_previstos():
    m = NGramaKN(2, V).treinar(_docs())
    saida = m.gerar([5], 50, np.random.default_rng(0))
    assert all(w not in FORA and w != EOS for w in saida)


def test_logp_janelas_reinicia_o_contexto_e_soma_os_nats():
    docs = _docs()
    m = NGramaKN(3, V).treinar(docs[:30])
    d = docs[0] * 4
    assert -m.logp_janelas(d, 10_000).sum() == pytest.approx(m.nats(d))
    lp = m.logp_janelas(d, 8)
    assert len(lp) == len(d) + 1
    assert lp[8] == pytest.approx(np.log(m.prob((BOS, BOS), d[8])))

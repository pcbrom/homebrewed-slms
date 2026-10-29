"""Blocking gates G0 and G1 of the formulation decision, run on the package code in float64.

G0: with L <= 6 and a tiny vocabulary, enumerating the L! reveal orders, (i) the number-of-masks
NELBO equals the average over orders of -log prod p (Ou et al. 2024, Theorem 2), (ii) every
order defines a normalised distribution, (iii) NELBO >= -log E_orders[prod p], and two injected
defects (mask logit kept, target visible to its own prediction) break normalisation.
G1: with exact conditionals of a known document distribution, the autoregressive NLL and the
diffusion NELBO coincide once padding is predicted, and giving the diffusion the length for free
lowers its loss by the entropy of the length.
"""
import itertools
import math

import pytest
import torch

from slm.avaliacao import avaliar
from slm.janelas import janelar
from slm.modelo import ConfigRede, Transformer
from slm.perda import perda_ar, perda_difusao
from slm.tokenizador import ESPECIAIS

MASK, BOS, EOS = ESPECIAIS["[MASK]"], ESPECIAIS["[BOS]"], ESPECIAIS["[EOS]"]
TOL = 1e-9
V, L = 6, 4
SIMBOLOS = [t for t in range(V) if t != MASK]


@pytest.fixture(scope="module")
def rede():
    torch.manual_seed(0)
    m = Transformer(ConfigRede(vocab_size=V, L=L, n_layer=2, d=16, n_head=2)).double().eval()
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(20.0)  # sharpen the random network so conditionals are far from uniform
    return m


def _todas_sequencias():
    return torch.tensor(list(itertools.product(SIMBOLOS, repeat=L)), dtype=torch.long)


@torch.no_grad()
def _logp_ordens(modelo, X, defeito_alvo_visivel=False):
    """log prod_k p(x_{sigma_k} | revealed) for every order sigma, shape (L!, N)."""
    saida = []
    for ordem in itertools.permutations(range(L)):
        lp = torch.zeros(X.shape[0], dtype=torch.float64)
        visivel = torch.zeros(L, dtype=torch.bool)
        for pos in ordem:
            ver = visivel.clone()
            if defeito_alvo_visivel:
                ver[pos] = True
            z = torch.where(ver, X, torch.full_like(X, MASK))
            lp += torch.log_softmax(modelo(z, False).double(), -1)[:, pos].gather(1, X[:, pos:pos + 1]).squeeze(1)
            visivel[pos] = True
        saida.append(lp)
    return torch.stack(saida)


@torch.no_grad()
def _nelbo_exata(modelo, X):
    """sum_l F_l / l by enumeration of every subset, using the package diffusion loss."""
    tot = torch.zeros(X.shape[0], dtype=torch.float64)
    for l in range(1, L + 1):
        subs = list(itertools.combinations(range(L), l))
        acc = torch.zeros(X.shape[0], dtype=torch.float64)
        for S in subs:
            m = torch.zeros(X.shape[0], L, dtype=torch.bool)
            m[:, list(S)] = True
            acc += perda_difusao(modelo, X, m)[1].double()  # (L / l) sum_{i in S} CE
        tot += acc / len(subs) / L
    return tot


def test_g0_nelbo_igual_media_das_ordens(rede):
    X = _todas_sequencias()
    ordens = _logp_ordens(rede, X)
    assert torch.allclose(_nelbo_exata(rede, X), -ordens.mean(0), atol=TOL, rtol=0)


def test_g0_normalizacao_e_limite(rede):
    X = _todas_sequencias()
    ordens = _logp_ordens(rede, X)
    p = ordens.exp().mean(0)
    assert abs(float(p.sum()) - 1.0) <= TOL
    assert bool((_nelbo_exata(rede, X) >= -torch.log(p) - TOL).all())


def test_g0_defeito_logit_de_mask_reprova(rede):
    class SemBloqueio(torch.nn.Module):
        def forward(self, ids, causal):
            m = rede
            x = m.emb(ids) + m.pos[: ids.shape[1]]
            for b in m.blocos:
                x = b(x, causal)
            return m.ln(x) @ m.emb.weight.T
    X = _todas_sequencias()
    p = _logp_ordens(SemBloqueio(), X).exp().mean(0)
    assert abs(float(p.sum()) - 1.0) > 1e-3


def test_g0_defeito_alvo_visivel_reprova(rede):
    X = _todas_sequencias()
    p = _logp_ordens(rede, X, defeito_alvo_visivel=True).exp().mean(0)
    assert abs(float(p.sum()) - 1.0) > 1e-3


class Oraculo(torch.nn.Module):
    """Exact conditionals of a known distribution over single-window documents."""

    def __init__(self, janelas, probs):
        super().__init__()
        self.J, self.P = janelas, probs

    def forward(self, ids, causal):
        B, T = ids.shape
        out = torch.full((B, T, V), float("-inf"), dtype=torch.float64)
        for b in range(B):
            for i in range(T):
                if causal:
                    pref = ids[b, 1:i + 1]
                    ok = (self.J[:, :i] == pref).all(1)
                else:
                    vis = ids[b] != MASK
                    ok = (self.J[:, vis] == ids[b, vis]).all(1)
                if not ok.any():
                    continue
                massa = torch.zeros(V, dtype=torch.float64).index_add_(0, self.J[ok, i], self.P[ok])
                out[b, i] = torch.log(massa / massa.sum())
        return out


def _distribuicao():
    docs = [list(s) for n in range(L) for s in itertools.product([4, 5], repeat=n)]
    g = torch.Generator().manual_seed(1)
    probs = torch.rand(len(docs), generator=g, dtype=torch.float64)
    return docs, probs / probs.sum()


def _nelbo_oraculo(modelo, W, livres=None):
    tot = torch.zeros(W.shape[0], dtype=torch.float64)
    for b in range(W.shape[0]):
        pos = livres[b] if livres is not None else list(range(L))
        for l in range(1, len(pos) + 1):
            subs = list(itertools.combinations(pos, l))
            acc = 0.0
            for S in subs:
                m = torch.zeros(1, L, dtype=torch.bool)
                m[0, list(S)] = True
                acc += float(perda_difusao(modelo, W[b:b + 1], m)[1]) * l / L
            tot[b] += acc / len(subs) / l
    return tot


def test_g1_preenchimento_fecha_o_vazamento_de_comprimento():
    docs, probs = _distribuicao()
    W, doc = janelar(docs, L)
    assert W.shape[0] == len(docs) and torch.equal(doc, torch.arange(len(docs)))
    oraculo = Oraculo(W, probs)
    nll_ar = perda_ar(oraculo, W)[1].double()
    nelbo = _nelbo_oraculo(oraculo, W)
    assert float((probs * (nll_ar - nelbo)).sum()) <= TOL
    assert torch.allclose(nll_ar, -torch.log(probs), atol=TOL, rtol=0)
    # leak: a diffusion told the length models only the content, conditioned on that length
    comprimentos = torch.tensor([len(d) for d in docs])
    p_n = torch.zeros(L, dtype=torch.float64).index_add_(0, comprimentos, probs)
    sem_custo = torch.zeros(len(docs), dtype=torch.float64)
    for n in comprimentos.unique().tolist():
        sel = (comprimentos == n).nonzero().squeeze(1)
        if n == 0:
            continue
        cond = Oraculo(W[sel], probs[sel] / p_n[n])
        sem_custo[sel] = _nelbo_oraculo(cond, W[sel], [list(range(n))] * len(sel))
    H_n = float(-(p_n[p_n > 0] * torch.log(p_n[p_n > 0])).sum())
    assert abs(float((probs * (nelbo - sem_custo)).sum()) - H_n) <= TOL


def test_g1_soma_de_bytes_do_iterador():
    docs, probs = _distribuicao()
    W, _ = janelar(docs, L)
    documentos = [{"doc_id": f"d{i}", "grupo": "g", "ids": d, "bytes": len(d) + 3} for i, d in enumerate(docs)]
    res = avaliar(Oraculo(W, probs), documentos, "autoregressivo", L)
    assert sum(r["bytes"] for r in res["documentos"]) == sum(d["bytes"] for d in documentos)
    assert math.isfinite(res["bpb"])

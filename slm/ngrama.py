"""N-gram language model with interpolated modified Kneser-Ney smoothing, the floor reference.

Chen and Goodman (1999): the highest order uses raw counts, every lower order uses continuation
counts (how many distinct tokens precede the n-gram), each order discounts counts of 1, 2 and 3 or
more by D1, D2 and D3+ estimated from its count-of-counts, and the discounted mass goes to the next
lower order. The lowest order interpolates with the uniform distribution over the tokens that can
be predicted, so every conditional distribution sums to 1 exactly.

The model works on the ids of the experiment's tokenizer. A document is preceded by n-1 [BOS]
tokens and followed by [EOS], which is predicted, as the Transformer predicts it. [PAD], [BOS] and
[MASK] are never predicted. Counting is vectorised with numpy; scoring walks the orders per position.
"""
from __future__ import annotations

import numpy as np

from slm.tokenizador import ESPECIAIS

BOS, EOS = ESPECIAIS["[BOS]"], ESPECIAIS["[EOS]"]
FORA = (ESPECIAIS["[PAD]"], BOS, ESPECIAIS["[MASK]"])
D_RESERVA = (0.5, 1.0, 1.5)


def descontos(contagens: np.ndarray) -> tuple[float, float, float]:
    """D1, D2, D3+ from the count-of-counts n1..n4; reserve values when a count is missing or a
    discount leaves its range (0, k), as happens in very small corpora."""
    n = [int(np.sum(contagens == k)) for k in (1, 2, 3, 4)]
    if min(n) == 0:
        return D_RESERVA
    y = n[0] / (n[0] + 2 * n[1])
    d = (1 - 2 * y * n[1] / n[0], 2 - 3 * y * n[2] / n[1], 3 - 4 * y * n[3] / n[2])
    return d if all(0 < dk < k for dk, k in zip(d, (1, 2, 3))) else D_RESERVA


def _tabela(gramas: np.ndarray, contagens: np.ndarray) -> tuple[dict, dict]:
    """Discounted counts per n-gram and, per context, the total and the back-off weight numerator."""
    d = descontos(contagens)
    desc = np.array([0.0, d[0], d[1], d[2]])[np.minimum(contagens, 3)]
    if gramas.shape[1] == 1:
        ctx_ids = np.zeros(len(gramas), dtype=np.int64)
        ctx_unicos = np.zeros((1, 0), dtype=gramas.dtype)
    else:
        ctx_unicos, ctx_ids = np.unique(gramas[:, :-1], axis=0, return_inverse=True)
        ctx_ids = ctx_ids.ravel()
    total = np.bincount(ctx_ids, weights=contagens)
    massa = np.bincount(ctx_ids, weights=desc)
    contextos = {tuple(c): (t, m) for c, t, m in zip(ctx_unicos.tolist(), total, massa)}
    ngramas = {tuple(g): c - dk for g, c, dk in zip(gramas.tolist(), contagens, desc)}
    return ngramas, contextos


class NGramaKN:
    def __init__(self, ordem: int, vocab_size: int):
        if ordem < 1:
            raise ValueError("ordem deve ser pelo menos 1")
        self.ordem, self.V = ordem, vocab_size
        self.previstos = np.setdiff1d(np.arange(vocab_size), FORA)

    def treinar(self, documentos: list[list[int]]) -> "NGramaKN":
        n = self.ordem
        seqs = [np.array([BOS] * (n - 1) + list(d) + [EOS], dtype=np.int64) for d in documentos]
        janelas = np.concatenate([np.lib.stride_tricks.sliding_window_view(s, n) for s in seqs if len(s) >= n])
        gramas, contagens = np.unique(janelas, axis=0, return_counts=True)
        self.tabelas = [None] * (n + 1)
        self.tabelas[n] = _tabela(gramas, contagens)
        for m in range(n - 1, 0, -1):
            # continuation count of (h', w): distinct left extensions among the order m+1 types
            gramas, contagens = np.unique(gramas[:, 1:], axis=0, return_counts=True)
            self.tabelas[m] = _tabela(gramas, contagens)
        return self

    def prob(self, contexto: tuple, w: int) -> float:
        """P(w | last n-1 tokens of the context), interpolated from the lowest order up."""
        p = 1.0 / len(self.previstos)
        for m in range(1, self.ordem + 1):
            h = tuple(contexto[len(contexto) - m + 1:]) if m > 1 else ()
            ngramas, contextos = self.tabelas[m]
            if h not in contextos:
                continue
            total, massa = contextos[h]
            p = ngramas.get(h + (w,), 0.0) / total + massa / total * p
        return p

    def distribuicao(self, contexto: tuple) -> np.ndarray:
        """Full conditional distribution over the vocabulary, zero on the tokens never predicted."""
        p = np.zeros(self.V)
        p[self.previstos] = [self.prob(contexto, int(w)) for w in self.previstos]
        return p

    def nats(self, ids: list[int], com_eos: bool = True) -> float:
        """Negative log-likelihood in nats of a document, [EOS] included when com_eos."""
        seq = [BOS] * (self.ordem - 1) + list(ids) + ([EOS] if com_eos else [])
        return -sum(np.log(self.prob(tuple(seq[i - self.ordem + 1:i]), seq[i]))
                    for i in range(self.ordem - 1, len(seq)))

    def gerar(self, prefixo: list[int], n: int, rng: np.random.Generator) -> list[int]:
        """Ancestral sampling of n tokens after the prefix, stopping at [EOS]."""
        seq = [BOS] * (self.ordem - 1) + list(prefixo)
        saida = []
        for _ in range(n):
            p = self.distribuicao(tuple(seq[len(seq) - self.ordem + 1:]))
            w = int(rng.choice(self.V, p=p / p.sum()))
            if w == EOS:
                break
            seq.append(w)
            saida.append(w)
        return saida

    def logp_janelas(self, ids: list[int], L: int) -> np.ndarray:
        """Log-probability of every token and of [EOS] with the context restarting every L positions,
        the windows of the Transformer evaluation."""
        seq = list(ids) + [EOS]
        saida = []
        for i in range(0, len(seq), L):
            pedaco = [BOS] * (self.ordem - 1) + seq[i:i + L]
            saida += [np.log(self.prob(tuple(pedaco[j - self.ordem + 1:j]), pedaco[j]))
                      for j in range(self.ordem - 1, len(pedaco))]
        return np.array(saida)

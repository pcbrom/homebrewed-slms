"""Paired bootstrap, seed count, TOST and Clopper-Pearson (box 9).

Definitions follow docs/SPEC.md, section 9, and docs/decisao_orcamento.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import stats

LN2 = float(np.log(2.0))

TEXTO_ABAIXO = "difusão melhor em verossimilhança"
TEXTO_ACIMA = (
    "o autoregressivo supera o limite da difusão, "
    "sem conclusão sobre a verossimilhança verdadeira dela"
)
TEXTO_CONTEM = "inconclusivo"


# --------------------------------------------------------------------------
# Bits per byte
# --------------------------------------------------------------------------

def bpb(nats, bytes_) -> float:
    """Bits per byte as a ratio of sums: sum(nats) / (ln 2 * sum(bytes))."""
    total_bytes = float(np.sum(bytes_))
    if total_bytes <= 0:
        raise ValueError("soma de bytes deve ser positiva")
    return float(np.sum(nats)) / (LN2 * total_bytes)


def bpb_documentos(documentos: Sequence[Mapping], sem_nomes: bool = False) -> float:
    """Bits per byte of a document list in the `avaliacao.json` format."""
    chave_n, chave_b = ("nats_sem_nomes", "bytes_sem_nomes") if sem_nomes else ("nats", "bytes")
    nats = np.fromiter((d[chave_n] for d in documentos), dtype=np.float64, count=len(documentos))
    bytes_ = np.fromiter((d[chave_b] for d in documentos), dtype=np.float64, count=len(documentos))
    return bpb(nats, bytes_)


def bpb_sem_nomes(documentos: Sequence[Mapping]) -> float:
    """Bits per byte excluding proper-name spans (SPEC, section 8)."""
    return bpb_documentos(documentos, sem_nomes=True)


@dataclass(frozen=True)
class DadosPareados:
    """Per-seed, per-document nats of both modes, aligned on the same documents."""

    nats_dif: np.ndarray  # (S, D)
    nats_ar: np.ndarray  # (S, D)
    bytes_: np.ndarray  # (D,)
    grupos: np.ndarray  # (D,)
    doc_ids: tuple


def _documentos(avaliacao) -> Sequence[Mapping]:
    return avaliacao["documentos"] if isinstance(avaliacao, Mapping) else avaliacao


def dados_pareados(
    avaliacoes_dif: Sequence,
    avaliacoes_ar: Sequence,
    sem_nomes: bool = False,
) -> DadosPareados:
    """Align `avaliacao.json` contents (one per seed, same seed order in both modes).

    Each element is either the parsed `avaliacao.json` or its `documentos` list.
    Documents are matched by `doc_id`; bytes and groups must agree across runs.
    """
    if len(avaliacoes_dif) != len(avaliacoes_ar) or not avaliacoes_dif:
        raise ValueError("as duas listas devem ter o mesmo número positivo de sementes")
    chave_n, chave_b = ("nats_sem_nomes", "bytes_sem_nomes") if sem_nomes else ("nats", "bytes")
    ref = _documentos(avaliacoes_dif[0])
    doc_ids = tuple(d["doc_id"] for d in ref)
    pos = {doc: i for i, doc in enumerate(doc_ids)}
    if len(pos) != len(doc_ids):
        raise ValueError("doc_id repetido")
    bytes_ = np.array([d[chave_b] for d in ref], dtype=np.float64)
    grupos = np.array([d["grupo"] for d in ref])

    def matriz(avaliacoes):
        out = np.empty((len(avaliacoes), len(doc_ids)), dtype=np.float64)
        for s, av in enumerate(avaliacoes):
            docs = _documentos(av)
            if len(docs) != len(doc_ids):
                raise ValueError("conjuntos de documentos diferentes entre corridas")
            for d in docs:
                i = pos[d["doc_id"]]
                if d[chave_b] != bytes_[i] or d["grupo"] != grupos[i]:
                    raise ValueError(f"bytes ou grupo divergente em {d['doc_id']}")
                out[s, i] = d[chave_n]
        return out

    return DadosPareados(matriz(avaliacoes_dif), matriz(avaliacoes_ar), bytes_, grupos, doc_ids)


# --------------------------------------------------------------------------
# Two-level bootstrap
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class IntervaloBootstrap:
    estimativa: float
    inferior: float
    superior: float
    deltas_sementes: np.ndarray  # (S,) observed Delta_s
    replicas: np.ndarray | None = None  # (R,) bootstrap means


def _somas_por_grupo(nats_dif, nats_ar, bytes_, grupos):
    _, idx = np.unique(np.asarray(grupos), return_inverse=True)
    idx = idx.ravel()
    g = int(idx.max()) + 1
    ind = np.zeros((idx.size, g))
    ind[np.arange(idx.size), idx] = 1.0
    return nats_dif @ ind, nats_ar @ ind, np.asarray(bytes_, dtype=np.float64) @ ind


def _deltas_reamostrados(n_dif_g, n_ar_g, b_g, rng, r):
    """Per-replicate, per-seed Delta under group resampling: shape (R, S)."""
    g = b_g.size
    cont = rng.multinomial(g, np.full(g, 1.0 / g), size=r).astype(np.float64)  # (R, G)
    den = LN2 * (cont @ b_g)  # (R,)
    return (cont @ n_dif_g.T - cont @ n_ar_g.T) / den[:, None]


def _pesos_sementes(s, rng, r):
    return rng.multinomial(s, np.full(s, 1.0 / s), size=r).astype(np.float64) / s


def bootstrap_delta(
    nats_dif,
    nats_ar,
    bytes_,
    grupos,
    R: int = 10_000,
    semente: int = 0,
    confianca: float = 0.95,
    guardar_replicas: bool = False,
) -> IntervaloBootstrap:
    """Two-level percentile bootstrap for Delta = bpb_dif - bpb_AR, paired by seed.

    `nats_dif` and `nats_ar` have shape (S, D), seed s in row s of both;
    `bytes_` and `grupos` have shape (D,). Groups (works) and seeds are
    resampled with replacement, independently; each replicate is the mean over
    resampled seeds of the per-seed difference of ratio-of-sums bpb.
    Cost O(R * G * S) after a single O(S * D) aggregation per group.
    """
    nats_dif = np.atleast_2d(np.asarray(nats_dif, dtype=np.float64))
    nats_ar = np.atleast_2d(np.asarray(nats_ar, dtype=np.float64))
    if nats_dif.shape != nats_ar.shape:
        raise ValueError("nats_dif e nats_ar devem ter a mesma forma (S, D)")
    n_dif_g, n_ar_g, b_g = _somas_por_grupo(nats_dif, nats_ar, bytes_, grupos)
    deltas = (n_dif_g.sum(1) - n_ar_g.sum(1)) / (LN2 * b_g.sum())
    rng = np.random.default_rng(semente)
    d_rs = _deltas_reamostrados(n_dif_g, n_ar_g, b_g, rng, R)
    w = _pesos_sementes(deltas.size, rng, R)
    rep = np.einsum("rs,rs->r", w, d_rs)
    a = (1.0 - confianca) / 2.0
    lo, hi = np.quantile(rep, [a, 1.0 - a])
    return IntervaloBootstrap(
        float(deltas.mean()), float(lo), float(hi), deltas, rep if guardar_replicas else None
    )


def bootstrap_pareado(dados: DadosPareados, **kwargs) -> IntervaloBootstrap:
    """`bootstrap_delta` applied to aligned `DadosPareados`."""
    return bootstrap_delta(dados.nats_dif, dados.nats_ar, dados.bytes_, dados.grupos, **kwargs)


@dataclass(frozen=True)
class Cobertura:
    cobertura: float
    n_nulos: int
    contem_zero: np.ndarray  # (n_nulos,) bool
    aprovada: bool


def cobertura_troca_sinal(
    nats_dif,
    nats_ar,
    bytes_,
    grupos,
    n_nulos: int = 1000,
    R: int = 10_000,
    semente: int = 0,
    confianca: float = 0.95,
    minimo: float = 0.93,
) -> Cobertura:
    """Coverage of the bootstrap interval under sign-flip null sets (SPEC, section 9).

    Each null set multiplies the paired difference of each seed by an
    independent random sign, which equals swapping the two modes of that seed
    at every document. The bootstrap is redrawn independently for each set;
    coverage is the fraction of null intervals that contain zero.
    """
    nats_dif = np.atleast_2d(np.asarray(nats_dif, dtype=np.float64))
    nats_ar = np.atleast_2d(np.asarray(nats_ar, dtype=np.float64))
    n_dif_g, n_ar_g, b_g = _somas_por_grupo(nats_dif, nats_ar, bytes_, grupos)
    s = n_dif_g.shape[0]
    rng = np.random.default_rng(semente)
    sinais = rng.choice(np.array([-1.0, 1.0]), size=(n_nulos, s))
    a = (1.0 - confianca) / 2.0
    contem = np.empty(n_nulos, dtype=bool)
    # loop over null sets keeps memory at O(R * S); the inner work is vectorized
    for j in range(n_nulos):
        d_rs = _deltas_reamostrados(n_dif_g, n_ar_g, b_g, rng, R) * sinais[j]
        rep = np.einsum("rs,rs->r", _pesos_sementes(s, rng, R), d_rs)
        lo, hi = np.quantile(rep, [a, 1.0 - a])
        contem[j] = lo <= 0.0 <= hi
    cob = float(contem.mean())
    return Cobertura(cob, n_nulos, contem, cob >= minimo)


def teste_aleatorizacao(deltas, n_perm: int = 100_000, semente: int = 0) -> float:
    """Two-sided sign-flip randomization p-value for the mean of per-seed deltas.

    Exact enumeration when 2**S <= n_perm, Monte Carlo otherwise.
    Fallback of docs/decisao_orcamento.md when coverage fails.
    """
    d = np.asarray(deltas, dtype=np.float64)
    s = d.size
    obs = abs(d.mean())
    if 2**s <= n_perm:
        bits = (np.arange(2**s)[:, None] >> np.arange(s)) & 1
        sinais = 1.0 - 2.0 * bits
        return float(np.mean(np.abs(sinais @ d) / s >= obs - 1e-15))
    rng = np.random.default_rng(semente)
    sinais = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, s))
    return float((1 + np.sum(np.abs(sinais @ d) / s >= obs - 1e-15)) / (n_perm + 1))


# --------------------------------------------------------------------------
# Seed count and minimum margin
# --------------------------------------------------------------------------

def fator_sigma_sup(m: int = 10, confianca: float = 0.95) -> float:
    """sqrt((m-1)/chi2_{(1-conf)/2; m-1}): upper confidence factor for sigma."""
    return float(np.sqrt((m - 1) / stats.chi2.ppf((1.0 - confianca) / 2.0, m - 1)))


def sigma_sup(sigma_hat: float, m: int = 10, confianca: float = 0.95) -> float:
    """Upper limit of the confidence interval for sigma, from m pilot seeds."""
    return float(sigma_hat) * fator_sigma_sup(m, confianca)


@dataclass(frozen=True)
class NumeroSementes:
    n: int | None  # None when not resolvable within the cap
    resolvivel: bool
    sigma_sup: float
    teto: int


def numero_sementes(
    sigma_hat: float,
    delta_min: float,
    m: int = 10,
    teto: int = 30,
    alfa: float = 0.05,
    poder: float = 0.8,
) -> NumeroSementes:
    """Smallest n >= 2 with n >= (t_{1-alfa/2;n-1} + t_{poder;n-1})^2 (sigma_sup/delta_min)^2."""
    if delta_min <= 0:
        raise ValueError("delta_min deve ser positivo")
    s_sup = sigma_sup(sigma_hat, m)
    n = np.arange(2, teto + 1)
    gl = n - 1
    exigido = (stats.t.ppf(1.0 - alfa / 2.0, gl) + stats.t.ppf(poder, gl)) ** 2 * (s_sup / delta_min) ** 2
    ok = np.flatnonzero(n >= exigido)
    if ok.size == 0:
        return NumeroSementes(None, False, s_sup, teto)
    return NumeroSementes(int(n[ok[0]]), True, s_sup, teto)


def desvio_piloto(deltas) -> float:
    """Sample standard deviation (ddof=1) of the pilot per-seed deltas."""
    return float(np.std(np.asarray(deltas, dtype=np.float64), ddof=1))


@dataclass(frozen=True)
class MargemMinima:
    valor: float
    erro_padrao: float  # nan with a single seed
    positiva: bool  # False triggers the 4 x 256 network (gate G3)


def delta_min(bpb_ar_4, bpb_ar_16) -> MargemMinima:
    """Delta_min = (bpb_AR^(4) - bpb_AR^(16)) / 4, averaged over paired seeds."""
    d = (np.atleast_1d(np.asarray(bpb_ar_4, dtype=np.float64))
         - np.atleast_1d(np.asarray(bpb_ar_16, dtype=np.float64))) / 4.0
    v = float(d.mean())
    ep = float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else float("nan")
    return MargemMinima(v, ep, v > 0)


# --------------------------------------------------------------------------
# TOST
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ResultadoTOST:
    media: float
    p_inferior: float  # H0: mu <= -margem
    p_superior: float  # H0: mu >= +margem
    p: float
    equivalente: bool


def tost_pareado(x, y, margem: float, alfa: float = 0.05) -> ResultadoTOST:
    """Paired TOST on d = x - y with equivalence margin (-margem, +margem).

    Equivalence is declared when both one-sided t tests reject at `alfa`.
    With zero variance the statistics are +-inf, so identical data inside
    the margin is declared equivalent.
    """
    d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    n = d.size
    if n < 2:
        raise ValueError("TOST exige pelo menos 2 pares")
    media = float(d.mean())
    ep = np.float64(d.std(ddof=1) / np.sqrt(n))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_inf = (media + margem) / ep
        t_sup = (media - margem) / ep
    p_inf = float(stats.t.sf(t_inf, n - 1))
    p_sup = float(stats.t.cdf(t_sup, n - 1))
    p = max(p_inf, p_sup)
    return ResultadoTOST(media, p_inf, p_sup, p, bool(np.isfinite(p) and p < alfa))


# --------------------------------------------------------------------------
# Clopper-Pearson
# --------------------------------------------------------------------------

def clopper_pearson_inferior(x, n, confianca: float = 0.95):
    """One-sided lower bound: quantile (1-conf) of Beta(x, n-x+1); 0 when x = 0."""
    x = np.asarray(x, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        lim = stats.beta.ppf(1.0 - confianca, x, n - x + 1)
    out = np.where(x > 0, lim, 0.0)
    return float(out) if out.ndim == 0 else out


def clopper_pearson(x, n, confianca: float = 0.95):
    """Two-sided exact interval (lower, upper) at the given confidence."""
    x = np.asarray(x, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    a = (1.0 - confianca) / 2.0
    with np.errstate(invalid="ignore"):
        lo = np.where(x > 0, stats.beta.ppf(a, x, n - x + 1), 0.0)
        hi = np.where(x < n, stats.beta.ppf(1.0 - a, x + 1, n - x), 1.0)
    if lo.ndim == 0:
        return float(lo), float(hi)
    return lo, hi


# --------------------------------------------------------------------------
# Krippendorff's alpha
# --------------------------------------------------------------------------

def matriz_coincidencias(dados):
    """Values and coincidence matrix from a (raters, units) array with nan as missing.

    Units with fewer than two pairable values are dropped (Krippendorff, 2011).
    """
    dados = np.asarray(dados, dtype=np.float64)
    if dados.ndim != 2:
        raise ValueError("dados devem ter forma (avaliadores, unidades)")
    valido = ~np.isnan(dados)
    valores, inv = np.unique(dados[valido], return_inverse=True)
    k = valores.size
    unid = np.broadcast_to(np.arange(dados.shape[1]), dados.shape)[valido]
    cont = np.zeros((dados.shape[1], k))
    np.add.at(cont, (unid, inv.ravel()), 1.0)
    m_u = cont.sum(1)
    par = m_u >= 2
    cont, m_u = cont[par], m_u[par]
    w = cont / (m_u - 1.0)[:, None]
    o = cont.T @ w - np.diag(w.sum(0))
    return valores, o


def _delta2(metrica: str, valores, n_c):
    if metrica == "nominal":
        return 1.0 - np.eye(valores.size)
    if metrica == "intervalar":
        return (valores[:, None] - valores[None, :]) ** 2
    if metrica == "ordinal":
        acum = np.cumsum(n_c)
        i = np.arange(valores.size)
        lo, hi = np.minimum.outer(i, i), np.maximum.outer(i, i)
        soma = acum[hi] - acum[lo] + n_c[lo]  # sum of n_g for g in [lo, hi]
        return (soma - (n_c[:, None] + n_c[None, :]) / 2.0) ** 2
    raise ValueError(f"métrica desconhecida: {metrica}")


def alfa_krippendorff(dados, metrica: str = "ordinal") -> float:
    """Krippendorff's alpha from the coincidence matrix (Krippendorff, 2011).

    `dados` has shape (raters, units), with nan (or None) for missing values.
    Metrics: "ordinal" (default), "nominal", "intervalar". Returns nan when
    expected disagreement is zero (a single value used, alpha undefined).
    """
    valores, o = matriz_coincidencias(dados)
    n_c = o.sum(1)
    n = n_c.sum()
    if n < 2:
        return float("nan")
    d2 = _delta2(metrica, valores, n_c)
    d_obs = float((o * d2).sum())
    d_esp = float((np.outer(n_c, n_c) * d2).sum()) / (n - 1.0)
    if d_esp == 0.0:
        return float("nan")
    return 1.0 - d_obs / d_esp


# --------------------------------------------------------------------------
# Reading of the result
# --------------------------------------------------------------------------

def ler_resultado(inferior: float, superior: float) -> tuple[str, str]:
    """Classify the interval of Delta: ('abaixo'|'acima'|'contem_zero', SPEC text)."""
    if superior < 0.0:
        return "abaixo", TEXTO_ABAIXO
    if inferior > 0.0:
        return "acima", TEXTO_ACIMA
    return "contem_zero", TEXTO_CONTEM

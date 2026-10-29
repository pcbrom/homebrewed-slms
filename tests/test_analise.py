"""Calibration criteria, deletion curves and figures on synthetic data and a tiny network."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from slm import calibracao as cal  # noqa: E402
from slm import delecao, graficos  # noqa: E402
from slm.janelas import janelar  # noqa: E402
from slm.modelo import ConfigRede, Transformer  # noqa: E402
from slm.tokenizador import BPEDidatico  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]


def test_temperatura_newton_recupera_t():
    g = torch.Generator().manual_seed(0)
    Z = torch.randn(20000, 12, generator=g) * 3
    y = torch.multinomial(torch.softmax(Z / 2.5, 1), 1, generator=g).squeeze(1)
    assert abs(cal.temperatura_newton(Z, y) - 2.5) < 0.1


def test_ece_auc_seletiva_portao():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.2, 1, 50000)
    acerto = rng.uniform(size=conf.size) < conf
    assert cal.ece(conf, acerto) < 0.01
    assert cal.auc_roc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([True, True, False, False])) == 1.0
    s = cal.curva_seletiva(conf, acerto)
    assert s["cobertura"][0] == 1.0 and s["acuracia"][-1] >= s["acuracia"][0]
    p = cal.portao_beta(conf[:25000], acerto[:25000], conf[25000:], acerto[25000:], alvo=0.9)
    assert p["limiar"] is not None and p["acuracia_seletiva"] >= 0.85


def _modelo(V, L):
    torch.manual_seed(0)
    return Transformer(ConfigRede(vocab_size=V, L=L, n_layer=1, d=16, n_head=2)).eval()


@pytest.mark.parametrize("modo", ["autoregressivo", "difusao"])
def test_coletar_previsoes(modo):
    m = _modelo(12, 16)
    J = janelar([[4, 5, 6, 7, 8] * 7] * 4, 16)[0]
    Z, y = cal.coletar_previsoes(m, J, modo, n_max=50)
    assert Z.shape[0] == y.shape[0] <= 50 and Z.shape[1] == 12


def test_coletar_previsoes_sem_preenchimento():
    # 35 tokens + EOS in windows of 16: 48 positions, of which 12 are padding after the EOS
    m = _modelo(12, 16)
    J = janelar([[4, 5, 6, 7, 8] * 7], 16)[0]
    Z, y = cal.coletar_previsoes(m, J, "autoregressivo", n_max=10**6, sem_preenchimento=True)
    assert y.shape[0] == 36 and int((y == 2).sum()) == 1


@pytest.mark.parametrize("modo", ["autoregressivo", "difusao"])
def test_curva_de_delecao(modo):
    texto = (RAIZ / "docs" / "PROBLEMA.md").read_text(encoding="utf-8")
    tok = BPEDidatico().treinar([texto], 300)
    m = _modelo(tok.vocab_size, 64)
    its = delecao.itens([texto], tok, n_itens=4, n_palavras=12)
    r = delecao.analisar(m, modo, tok, its, L=64)
    for it in r["itens"]:
        assert len(it["curva_importancia"]) == 11 and it["curva_importancia"][0] == pytest.approx(1.0)
        assert all(0 <= v <= 1 for v in it["curva_importancia"] + it["curva_aleatoria"])
    assert 0 <= r["auc_importancia_media"] <= 1 and "rho" in r["spearman_p0_auc"]


def test_graficos_geram_png(tmp_path):
    rng = np.random.default_rng(0)
    modos = ("autoregressivo", "difusao")
    passos = {m: [{"passo": i, "perda": 5 - i / 50, "taxa": 1e-3, "escala_gradscaler": 65536.0, "passos_pulados": i // 40}
                  for i in range(0, 200, 10)] for m in modos}
    sent = [{"passo": p, "faixa_l": f, "tensor": "w", "erro_relativo_p95": 0.01, "underflow": 0.001}
            for p in (0, 100) for f in ("[0.0,0.01)", "[0.5,1.0)")]
    val = {m: {e: list(3 - np.log(e) / 5 + rng.normal(0, 0.02, 5)) for e in (1, 4, 16, 64, 256)} for m in modos}
    tv = {m: {"treino": [(e, 3 - np.log(e) / 3) for e in (1, 4, 16)], "validacao": [(e, 3 - np.log(e) / 5) for e in (1, 4, 16)]} for m in modos}
    conf = rng.uniform(0.1, 1, 3000)
    ac = rng.uniform(size=conf.size) < conf
    conf_d = dict(cal.confiabilidade(conf, ac), ece=cal.ece(conf, ac))
    figs = {
        "treino": graficos.treino(passos), "sentinela": graficos.sentinela(sent),
        "validacao": graficos.validacao_epocas(val), "treino_validacao": graficos.treino_validacao(tv),
        "delta": graficos.delta_por_obra({"a": (0.01, -0.01, 0.03), "b": (0.02, 0.0, 0.04)}, (0.015, 0.0, 0.03)),
        "nomes": graficos.com_sem_nomes({m: (2.1, 2.0) for m in modos}),
        "novidade": graficos.novidade_ngramas({"autoregressivo": list(np.linspace(0.01, 0.9, 10)), "texto humano": list(np.linspace(0.02, 0.95, 10))}),
        "descarte": graficos.descarte_copia({m: {e: (0.1, 0.05, 0.2) for e in (1, 16, 256)} for m in modos}),
        "entropia": graficos.entropia_genppl({m: [(0.8, 4.9, 30.0), (1.0, 5.2, 60.0)] for m in modos}),
        "niveis": graficos.niveis({"N2": {m: {e: (0.5, 0.3) for e in (1, 16, 256)} for m in modos}}),
        "confiabilidade": graficos.confiabilidade({"autoregressivo calibrado": conf_d}),
        "seletiva": graficos.seletiva({"autoregressivo": cal.curva_seletiva(conf, ac)}, alvo=0.9),
        "grade": graficos.sensibilidade_grade({m: rng.uniform(2, 3, (3, 3)) for m in modos}, [1e-4, 3e-4, 1e-3], [0.0, 0.1, 0.3], "taxa", "decaimento"),
        "linha": graficos.sensibilidade_linha({m: {k: list(rng.normal(2, 0.01, 4)) for k in (1, 4, 16)} for m in modos}, "K", "bpb"),
    }
    r = {"grade": list(delecao.GRADE), "curva_importancia_media": list(np.linspace(1, 0.2, 11)),
         "curva_aleatoria_media": list(np.linspace(1, 0.5, 11)), "auc_importancia_media": 0.5, "auc_aleatoria_media": 0.7,
         "itens": [{"p0": 0.5, "auc_importancia": 0.4}, {"p0": 0.8, "auc_importancia": 0.6}], "spearman_p0_auc": {"rho": 1.0}}
    figs["delecao"] = graficos.delecao({m: r for m in modos})
    for nome, fig in figs.items():
        assert graficos.salvar(fig, tmp_path / f"{nome}.png").stat().st_size > 1000

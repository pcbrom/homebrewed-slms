"""Training loop, sentinel, generation, configuration and pre-registration on CPU."""
import pytest
import torch

from slm.avaliacao import nats_por_janela
from slm.config import Config
from slm.mascaramento import MASK, mascara_de_l, sortear_l
from slm.geracao import gerar_ar, gerar_difusao, sortear
from slm.janelas import fracao_preenchimento, janelar
from slm.modelo import ConfigRede, Transformer
from slm.registro import RegistroCorrida, conferir_preregistro, preregistrar
from slm.treino import mascara_do_passo, sentinela, sondas_por_faixa, taxa, treinar

V, L = 12, 16


def _dados():
    padrao = [4, 5, 6, 7, 8, 9, 10, 11] * 6
    return janelar([padrao[i:i + 30] for i in range(8)], L)[0]


def _modelo(semente=0):
    torch.manual_seed(semente)
    return Transformer(ConfigRede(vocab_size=V, L=L, n_layer=2, d=32, n_head=4))


@pytest.mark.parametrize("modo", ["autoregressivo", "difusao"])
def test_treino_reduz_a_perda(modo, tmp_path):
    cfg = Config(modo=modo, vocab_size=V, L=L, n_layer=2, d=32, n_head=4, lote=8, epocas=150, lr=3e-3,
                 precisao="fp32", atencao="math", registro_cada=10)
    # identical windows: both modes can fit them in a few hundred steps
    dados, m = janelar([[4, 5, 6, 7, 8, 9, 10, 11, 10, 9, 8, 7, 6, 5]] * 16, L)[0], _modelo()
    antes = nats_por_janela(m.eval(), dados, modo, K=8).mean()
    res = treinar(cfg, m, dados, registro=RegistroCorrida(tmp_path))
    depois = nats_por_janela(m.eval(), dados, modo, K=8).mean()
    assert res["passos_aplicados"] > 0 and res["passos_pulados"] == 0
    assert len((tmp_path / "passos.jsonl").read_text().splitlines()) >= 2
    assert depois < 0.5 * antes


def test_mascara_do_passo_repetivel():
    cfg = Config(modo="difusao", vocab_size=V, L=L)
    assert torch.equal(mascara_do_passo(cfg, 7, 4, L), mascara_do_passo(cfg, 7, 4, L))
    assert not torch.equal(mascara_do_passo(cfg, 7, 4, L), mascara_do_passo(cfg, 8, 4, L))


def test_sortear_l_estratificado_e_mascara_exata():
    g = torch.Generator().manual_seed(0)
    l = sortear_l(64, L, g)
    assert l.min() >= 1 and l.max() <= L
    m = mascara_de_l(l, L, g)
    assert torch.equal(m.sum(1), l)


def test_cronograma_cosseno():
    cfg = Config(modo="difusao", vocab_size=V, lr=1e-3, lr_min_frac=0.1, aquecimento_frac=0.1)
    assert taxa(0, 100, cfg) < taxa(9, 100, cfg) <= 1e-3
    assert abs(taxa(99, 100, cfg) - 1e-4) < 2e-6


@pytest.mark.parametrize("modo", ["autoregressivo", "difusao"])
def test_sentinela_em_cpu(modo):
    m = _modelo()
    sondas = sondas_por_faixa(_dados(), modo, n=4, lotes=2)
    linhas = sentinela(m, sondas, modo, "cpu", escala=1.0, passo=0, dtype=torch.bfloat16)
    assert linhas and all(r["underflow"] >= 0 for r in linhas)
    assert any(r["erro_relativo_p95"] is not None for r in linhas)
    assert all(p.grad is None for p in m.parameters())


@pytest.mark.parametrize("modo", ["autoregressivo", "difusao"])
def test_sentinela_desliga_o_dropout(modo):
    torch.manual_seed(0)
    m = Transformer(ConfigRede(vocab_size=V, L=L, n_layer=2, d=32, n_head=4, dropout=0.5)).train()
    sondas = sondas_por_faixa(_dados(), modo, n=4, lotes=2)
    linhas = sentinela(m, sondas, modo, "cpu", escala=1.0, passo=0, dtype=None)
    assert all(r["erro_relativo_p95"] in (None, 0.0) for r in linhas)
    assert m.training


def test_geracao_sem_mask_e_reprodutivel():
    m = _modelo().eval()
    a = gerar_difusao(m, L, T_gen=8, n=3, semente=5)
    b = gerar_difusao(m, L, T_gen=8, n=3, semente=5)
    assert torch.equal(a, b) and not (a == MASK).any()
    c = gerar_ar(m, L, n=2, prefixo=[4, 5], semente=1)
    assert c.shape == (2, L) and c[:, :2].tolist() == [[4, 5], [4, 5]] and not (c == MASK).any()


def test_sorteio_float64_respeita_proibidos():
    logits = torch.zeros(1000, V)
    logits[:, MASK] = float("-inf")
    s = sortear(logits, torch.Generator().manual_seed(0))
    assert not (s == MASK).any() and len(s.unique()) == V - 1


def test_fracao_preenchimento():
    assert fracao_preenchimento([[4] * 3], 4) == 0.0
    assert fracao_preenchimento([[4] * 4], 4) == 3 / 8


def test_config_valida():
    with pytest.raises(ValueError):
        Config(modo="outro", vocab_size=V).validar()
    c = Config(modo="difusao", vocab_size=V).validar()
    assert Config.de_dicionario(c.dicionario()).hash() == c.hash()


def test_preregistro_imutavel(tmp_path):
    preregistrar(tmp_path, "p1", {"delta_min": 0.01})
    assert conferir_preregistro(tmp_path, "p1")
    with pytest.raises(FileExistsError):
        preregistrar(tmp_path, "p1", {"delta_min": 0.02})

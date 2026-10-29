"""Blind-batch helpers of scripts/m9.py: random-start excerpts and items spread over seeds."""
import importlib.util
import random
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("m9", Path(__file__).resolve().parents[1] / "scripts" / "m9.py")
m9 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m9)


def test_cortar_devolve_n_palavras_consecutivas_de_inicio_sorteado():
    texto = " ".join(f"p{i}" for i in range(100))
    inicios = set()
    for s in range(50):
        trecho = m9.cortar(texto, 10, random.Random(s)).split()
        assert len(trecho) == 10
        i = int(trecho[0][1:])
        assert trecho == [f"p{j}" for j in range(i, i + 10)]
        inicios.add(i)
    assert len(inicios) > 10 and 0 <= min(inicios) and max(inicios) <= 90
    assert m9.cortar("curto demais", 10, random.Random(0)) is None


def test_excertos_humanos_tem_o_comprimento_pedido():
    textos = [" ".join(f"t{k}w{i}" for i in range(200)) for k in range(5)]
    ex = m9.excertos_humanos(textos, 30, 60, semente=0)
    assert len(ex) == 30 and all(len(e.split()) == 60 for e in ex)
    assert ex == m9.excertos_humanos(textos, 30, 60, semente=0)


def test_escolher_distribui_igualmente_entre_sementes():
    amostras = [{"semente": s, "id": f"{s}-{j}"} for s in range(4) for j in range(5)]
    escolha = m9.escolher(amostras, 12, semente=0)
    assert len(escolha) == 12
    assert sorted(sum(r["semente"] == s for r in escolha) for s in range(4)) == [3, 3, 3, 3]
    assert len({r["id"] for r in escolha}) == 12


def test_escolher_recusa_semente_sem_amostras_suficientes():
    amostras = [{"semente": 0, "id": str(j)} for j in range(5)] + [{"semente": 1, "id": "x"}]
    with pytest.raises(SystemExit):
        m9.escolher(amostras, 12, semente=0)

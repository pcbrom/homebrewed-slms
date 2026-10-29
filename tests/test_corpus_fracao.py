from slm.corpus import fracao_grupos

DOCS = [{"doc_id": f"d{i}", "grupo": f"g{i % 7}", "bytes": 100 + 13 * i} for i in range(40)]


def test_fracoes_aninhadas_e_por_grupo_inteiro():
    total = sum(d["bytes"] for d in DOCS)
    anteriores = set()
    for fr in (0.125, 0.25, 0.5, 1.0):
        sel = fracao_grupos(DOCS, fr)
        ids = {d["doc_id"] for d in sel}
        assert anteriores <= ids
        assert sum(d["bytes"] for d in sel) >= fr * total
        grupos = {d["grupo"] for d in sel}
        assert ids == {d["doc_id"] for d in DOCS if d["grupo"] in grupos}
        anteriores = ids
    assert len(fracao_grupos(DOCS, 1.0)) == len(DOCS)

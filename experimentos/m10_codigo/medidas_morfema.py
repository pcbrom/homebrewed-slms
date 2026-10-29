#!/usr/bin/env python3
"""Density of the hand-designed morpheme rule (experimentos/m10_codigo/morfema.py), pure and mixed, against BPE.

Reuses medidas_codigo.medir with the morpheme function replaced by each variant of the rule.

    .venv/bin/python experimentos/m10_codigo/medidas_morfema.py
"""
import collections
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
SAIDA = RAIZ / "resultados" / "m10"
sys.path.insert(0, str(AQUI))
import medidas_codigo as mc  # noqa: E402
import morfema  # noqa: E402

EXEMPLO = ("Capitú tinha olhos de ressaca, disse-lhe elle n'aquella hora. As desconfianças cresceram "
           "rapidamente, e os sobrinhos contrariados entregavam-se á tristeza das conversações.")


def main():
    treino = [d["texto"] for d in mc.ler_documentos("A", "treino")]
    valid = [d["texto"] for d in mc.ler_documentos("A", "validacao")]
    B = sum(len(t.encode("utf-8")) for t in valid)
    palavras = collections.Counter(p for t in treino for p in mc.PAT.findall(t))
    topo = {p for p, _ in palavras.most_common(4096)}
    res = []
    for var in ("flexao", "derivacao", "prefixo"):
        mc.morfemas = lambda p, v=var: morfema.segmentar(p, v)
        print(f"\n[{var}]", " | ".join(u for p in mc.PAT.findall(EXEMPLO) for u in mc.morfemas(p)))
        for unidade, topo_u, rotulo in (("morfema", None, f"morfema {var}"), ("mista", topo, f"mista 4096 {var}")):
            for r in mc.medir(unidade, treino, valid, B, (2, 256, 4096), topo_u):
                if r["esquema"] == "huffman":
                    r["unidade"] = rotulo
                    res.append(r)
    (SAIDA / "medidas_morfema.json").write_text(json.dumps({"bytes_validacao": B, "linhas": res}, indent=1, ensure_ascii=False), encoding="utf-8")
    cab = ["unidade", "A", "inventario", "oov_valid", "simbolos_por_byte", "bits_nominais_por_byte", "xent_unidades_por_byte", "info_por_bit"]
    print("\n" + " | ".join(cab))
    for r in res:
        print(" | ".join(str(r[k]) for k in cab))


if __name__ == "__main__":
    main()

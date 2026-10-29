#!/usr/bin/env python3
"""Draw 72 summaries of the full 2019 to 2022 law list for the manual audit of the class rule (corpus gate G2).

Reads the Senado lists saved by construir_corpus_b.py in dados_brutos/B/ and writes
corpus/B/auditoria_classes.csv with the columns ementa, classe_prevista and
classe_correta. The last column stays empty and is filled by hand. The rule
passes with zero errors in 72, which puts the 95% Clopper-Pearson upper bound
below 5%.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from slm import dados as D  # noqa: E402

ANOS = (2019, 2020, 2021, 2022)
TIPO_ORDINARIA = "LEI-n"
N_AMOSTRA = 72
SEMENTE = 20261022


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--brutos", type=Path, default=RAIZ / "dados_brutos" / "B")
    ap.add_argument("--saida", type=Path, default=RAIZ / "corpus" / "B" / "auditoria_classes.csv")
    args = ap.parse_args()

    leis = {}
    for ano in ANOS:
        arq = args.brutos / f"lista_{ano}.xml"
        if not arq.exists():
            raise SystemExit(f"lista ausente: {arq}; rode antes scripts/construir_corpus_b.py")
        for d in ET.parse(arq).getroot().findall(".//documento"):
            if d.findtext("tipo") == TIPO_ORDINARIA:
                leis[int(d.findtext("numero"))] = " ".join((d.findtext("ementa") or "").split())
    numeros = sorted(leis)
    amostra = sorted(random.Random(SEMENTE).sample(numeros, N_AMOSTRA))
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    with open(args.saida, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ementa", "classe_prevista", "classe_correta"])
        for n in amostra:
            w.writerow([leis[n], D.classificar_ementa(leis[n]), ""])
    print(f"{N_AMOSTRA} ementas de {len(numeros)} leis, semente {SEMENTE}: {args.saida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()

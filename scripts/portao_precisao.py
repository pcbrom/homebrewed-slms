#!/usr/bin/env python3
"""Gate G2 of the precision decision: fp16 against fp32, paired by seed (docs/decisao_precisao.md).

Reads runs/calibracao-<modo>-<fp16|fp32>-s<k>/avaliacao.json, computes per mode the paired
difference of validation bits per byte (fp16 minus fp32) and its 95% interval by t. When
preregistro/m6_delta_min.json exists, the margin is m_prec = Delta_min / 4 and the gate passes in
a mode when the interval lies inside +-m_prec.

    python scripts/portao_precisao.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as student

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from slm.registro import conferir_preregistro  # noqa: E402

MODOS = ("autoregressivo", "difusao")


def bpb(run_id: str) -> float:
    return json.loads((RAIZ / "runs" / run_id / "avaliacao.json").read_text(encoding="utf-8"))["bpb"]


def main():
    sementes = sorted({int(p.name.rsplit("-s", 1)[1]) for p in (RAIZ / "runs").glob("calibracao-*-fp16-s*")})
    pre = RAIZ / "preregistro"
    m_prec = None
    if (pre / "m6_delta_min.json").exists():
        assert conferir_preregistro(pre, "m6_delta_min"), "m6_delta_min.json altered after freezing"
        m_prec = json.loads((pre / "m6_delta_min.json").read_text())["m_prec"]
    saida = {"sementes": sementes, "m_prec": m_prec}
    for modo in MODOS:
        d = np.array([bpb(f"calibracao-{modo}-fp16-s{s}") - bpb(f"calibracao-{modo}-fp32-s{s}") for s in sementes])
        meia = student.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        ic = [float(d.mean() - meia), float(d.mean() + meia)]
        saida[modo] = {"diferencas": d.round(4).tolist(), "media": round(float(d.mean()), 4),
                       "ic95": [round(x, 4) for x in ic],
                       "aprovado": None if m_prec is None else bool(-m_prec < ic[0] and ic[1] < m_prec)}
    print(json.dumps(saida, indent=2))


if __name__ == "__main__":
    main()

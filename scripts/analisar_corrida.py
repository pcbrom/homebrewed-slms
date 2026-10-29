#!/usr/bin/env python3
"""Calibration and deletion analyses of one trained run (CONTRATO_DADO section 7).

Reads runs/<run_id>/config.json and pesos.pt, fits the temperature on the validation partition,
writes calibracao.json (ECE, reliability, selective curve, ROC AUC, Beta gate at several targets)
and delecao.json (deletion curves on test passages).

    python scripts/analisar_corrida.py --run-id confirmatoria-difusao-e256-s0 --dispositivo cuda
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slm import calibracao as cal  # noqa: E402
from slm import delecao  # noqa: E402
from slm.config import Config  # noqa: E402
from slm.corpus import RAIZ, ler_documentos, pasta_tokenizador, tokenizar  # noqa: E402
from slm.janelas import janelar  # noqa: E402
from slm.modelo import ConfigRede, Transformer  # noqa: E402
from slm.registro import RegistroCorrida  # noqa: E402
from slm.tokenizador import carregar_tokenizador  # noqa: E402

CAMPOS = set(Config.__dataclass_fields__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--dispositivo", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-previsoes", type=int, default=50000)
    ap.add_argument("--n-itens-delecao", type=int, default=300)
    ap.add_argument("--alvos", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    ap.add_argument("--tokenizador", default=None, help="default: tokenizador/<corpus>")
    a = ap.parse_args()

    pasta = RAIZ / "runs" / a.run_id
    bruto = json.loads((pasta / "config.json").read_text(encoding="utf-8"))
    cfg = Config.de_dicionario({k: v for k, v in bruto.items() if k in CAMPOS})
    tok = carregar_tokenizador(Path(a.tokenizador) if a.tokenizador else pasta_tokenizador(cfg))
    modelo = Transformer(ConfigRede(vocab_size=cfg.vocab_size, L=cfg.L, n_layer=cfg.n_layer, d=cfg.d, n_head=cfg.n_head))
    modelo.load_state_dict(torch.load(pasta / "pesos.pt", map_location="cpu"))
    modelo.to(a.dispositivo).eval()

    validacao = tokenizar(ler_documentos(cfg.corpus, "validacao"), tok)
    jv, _ = janelar([d["ids"] for d in validacao], cfg.L)
    # windows in a fixed random order: the n_previsoes cap then spans every work, and the two halves
    # of the Beta gate are exchangeable instead of coming from different works
    jv = jv[torch.randperm(jv.shape[0], generator=torch.Generator().manual_seed(0))]
    Z, y = cal.coletar_previsoes(modelo, jv, cfg.modo, n_max=a.n_previsoes, dispositivo=a.dispositivo, sem_preenchimento=True)
    T = cal.temperatura_newton(Z, y)
    metade = Z.shape[0] // 2
    c1, a1 = cal.confianca_acerto(Z, y, 1.0)
    cT, aT = cal.confianca_acerto(Z, y, T)
    res = {"T": T, "n_previsoes": int(Z.shape[0]), "ordem_janelas": "permutada, semente 0", "preenchimento": "excluido, EOS do documento mantido", "ece_T1": cal.ece(c1, a1), "ece_calibrado": cal.ece(cT, aT),
           "auc_roc": cal.auc_roc(cT, aT),
           "confiabilidade": {"T1": cal.confiabilidade(c1, a1), "calibrado": cal.confiabilidade(cT, aT)},
           "seletiva": cal.curva_seletiva(cT, aT),
           "portao_beta": {str(alvo): cal.portao_beta(cT[:metade], aT[:metade], cT[metade:], aT[metade:], alvo)
                           for alvo in a.alvos}}
    reg = RegistroCorrida(pasta)
    reg.avaliacao(res, "calibracao.json")

    teste = [d["texto"] for d in ler_documentos(cfg.corpus, "teste")]
    its = delecao.itens(teste, tok, n_itens=a.n_itens_delecao)
    reg.avaliacao(delecao.analisar(modelo, cfg.modo, tok, its, cfg.L, T, dispositivo=a.dispositivo), "delecao.json")
    print(json.dumps({"run_id": a.run_id, "T": round(T, 4), "ece_calibrado": round(res["ece_calibrado"], 4),
                      "auc_roc": round(res["auc_roc"], 4)}, indent=2))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()

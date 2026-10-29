#!/usr/bin/env python3
"""Sanity checks of the typed decider of M14, run after its measurement (not preregistered).

1. Overfitting a small batch: 32 training texts, half of each class, one full batch per step for
   300 steps at each rate of the grid and at 1e-3. A decider that can learn reaches accuracy 1 on
   the batch; one that cannot points to a defect in the pipeline rather than in the task.
2. Loss curve of the protocol fit: all training items, 3 epochs, batch 16, at the rate chosen in
   M14, with the balanced accuracy on the training items and the share of each predicted class.
3. Share of each predicted class on the external set, from the stored choices of every run.

Steps 1 and 2 use the first seed of each mode; step 3 needs no GPU.

    python scripts/sanidade_decisor.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))
from m14 import TAXAS, checkpoints, itens, textos_decisor  # noqa: E402
from slm import decisor as D  # noqa: E402
from slm.modelo import ConfigRede, Transformer  # noqa: E402
from slm.tokenizador import carregar_tokenizador  # noqa: E402

PASTA = RAIZ / "resultados" / "m14_B"
N_LOTE, PASSOS = 32, 300
MARCAS = (1, 10, 50, 100, 200, 300)


def balanceada(y, pred):
    return float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))


def main():
    tok = carregar_tokenizador(RAIZ / "tokenizador" / "B")
    disp = "cuda" if torch.cuda.is_available() else "cpu"
    resumo = json.loads((PASTA / "decisor_resumo.json").read_text())
    esc = checkpoints("B")
    saida = {"descricao": __doc__.split("\n\n")[0], "lote_pequeno": {}, "protocolo": {}, "classes_externo": {}}
    for modo, e in esc.items():
        rid = e["run_ids"][0]
        pasta = RAIZ / "runs" / rid
        cfg = json.loads((pasta / "config.json").read_text(encoding="utf-8"))
        enc = Transformer(ConfigRede(vocab_size=cfg["vocab_size"], L=cfg["L"], n_layer=cfg["n_layer"], d=cfg["d"],
                                     n_head=cfg["n_head"], dropout=cfg["dropout"]))
        enc.load_state_dict(torch.load(pasta / "pesos.pt", map_location="cpu"))
        causal = modo == "autoregressivo"
        for t in ("t2", "t3"):
            tr = textos_decisor(t, itens("B", t, {"treino"}))
            s_tr, y = [tok.encode(x) for x, _, _ in tr], np.array([r for _, r, _ in tr])
            rng = np.random.default_rng(0)
            idx = np.concatenate([rng.choice(np.flatnonzero(y == c), N_LOTE // 2, replace=False) for c in (0, 1)])
            s_l, y_l = [s_tr[k] for k in idx], y[idx]
            for lr in (*TAXAS, 1e-3):
                perdas = []
                dec = D.ajustar(enc, causal, s_l, y_l, lr, cfg["L"], epocas=PASSOS, tam_lote=N_LOTE,
                                semente=cfg["semente"], dispositivo=disp, perdas=perdas)
                pred = D.logits(dec, s_l, cfg["L"], dispositivo=disp).argmax(1)
                saida["lote_pequeno"][f"{rid}:{t}:{lr:g}"] = {"perda": {m: perdas[m - 1] for m in MARCAS},
                                                              "acerto_no_lote": float((pred == y_l).mean())}
                print(rid, t, lr, saida["lote_pequeno"][f"{rid}:{t}:{lr:g}"], flush=True)
            lr = resumo[f"{rid}:{t}"]["taxa"]
            perdas = []
            dec = D.ajustar(enc, causal, s_tr, y, lr, cfg["L"], semente=cfg["semente"], dispositivo=disp, perdas=perdas)
            pred = D.logits(dec, s_tr, cfg["L"], dispositivo=disp).argmax(1)
            k = max(1, len(perdas) // 20)
            saida["protocolo"][f"{rid}:{t}"] = {"taxa": lr, "passos": len(perdas), "perda_media_por_trecho": [float(np.mean(perdas[i:i + k])) for i in range(0, len(perdas), k)],
                                                "acuracia_balanceada_treino": balanceada(y, pred), "fracao_prevista_1_treino": float(pred.mean()),
                                                "fracao_rotulo_1_treino": float(y.mean())}
            print(rid, t, {a: b for a, b in saida["protocolo"][f"{rid}:{t}"].items() if a != "perda_media_por_trecho"}, flush=True)
    for modo, e in esc.items():
        for t in ("t2", "t3"):
            fr = [np.mean([json.loads(l)["p1"] > 0.5 for l in open(PASTA / f"decisor-{rid}_{t}.jsonl")]) for rid in e["run_ids"]]
            saida["classes_externo"][f"{modo}:{t}"] = {"fracao_prevista_1_por_semente": [float(x) for x in fr]}
    (PASTA / "sanidade_decisor.json").write_text(json.dumps(saida, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

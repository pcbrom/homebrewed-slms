#!/usr/bin/env python3
"""Design probe of T3 (docs/decisao_avaliacao.md, section 3): items per adulteration family, the
13-token filter, and whether a TF-IDF logistic regression reading one text at a time tells original
from adulterated by grouped cross-validation on the training partition (near 0.5 means no surface
cue). Prints one example per family and writes resultados/sonda_t3.json.

    python scripts/sonda_t3.py [externo]          # default B_externo
"""
import difflib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from slm.corpus import ler_documentos  # noqa: E402
from slm.tarefas import itens_t3, sem_sobreposicao  # noqa: E402
from slm.tokenizador import carregar_tokenizador  # noqa: E402

MAX_TOKENS = 254


def diferenca(a, b):
    s = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return [(a[i1:i2], b[j1:j2]) for op, i1, i2, j1, j2 in s.get_opcodes() if op != "equal"][:2]


def main():
    tok = carregar_tokenizador(RAIZ / "tokenizador" / "B")
    cabe = lambda t: len(tok.encode(t)) <= MAX_TOKENS
    treino_ids = [tok.encode(d["texto"]) for d in ler_documentos("B", "treino")]
    saida = {}
    for nome in ("B", sys.argv[1] if len(sys.argv) > 1 else "B_externo"):
        its = itens_t3(ler_documentos(nome), cabe)
        r = {"itens": len(its), "por_familia": dict(Counter(i["familia"] for i in its)),
             "grupos": len({i["grupo"] for i in its}), "por_particao": dict(Counter(i["particao"] for i in its))}
        aval = [i for i in its if i["particao"] != "treino"]
        ok_o = sem_sobreposicao([tok.encode(i["original"]) for i in aval], treino_ids)
        ok_a = sem_sobreposicao([tok.encode(i["adulterado"]) for i in aval], treino_ids)
        fic = [i for i, o, a in zip(aval, ok_o, ok_a) if o and a]
        r["avaliacao_sem_sobreposicao_13"] = {"antes": len(aval), "depois": len(fic),
                                              "por_familia": dict(Counter(i["familia"] for i in fic)),
                                              "grupos": len({i["grupo"] for i in fic})}
        saida[nome] = r
        print(nome, json.dumps(r, ensure_ascii=False))
        if nome == "B":
            for f in sorted(r["por_familia"]):
                i = next(i for i in its if i["familia"] == f)
                print(f"  {f}: {diferenca(i['original'], i['adulterado'])}")
    its = [i for i in itens_t3(ler_documentos("B", "treino"), cabe)]
    X = [t for i in its for t in (i["original"], i["adulterado"])]
    y = np.array([r for _ in its for r in (0, 1)])
    g = np.array([i["grupo"] for i in its for _ in (0, 1)])
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(5).split(X, y, g):
        v = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2))
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(v.fit_transform([X[k] for k in tr]), y[tr])
        pred[te] = clf.predict(v.transform([X[k] for k in te]))
    saida["tfidf_texto_unico_cv_agrupada_treino_B"] = float(balanced_accuracy_score(y, pred))
    saida["tfidf_por_familia"] = {f: float(balanced_accuracy_score(y[m], pred[m])) for f in sorted({i["familia"] for i in its})
                                  for m in [np.array([i["familia"] == f for i in its for _ in (0, 1)])]}
    print("TF-IDF texto unico, CV agrupada no treino:", round(saida["tfidf_texto_unico_cv_agrupada_treino_B"], 3),
          {k: round(v, 3) for k, v in saida["tfidf_por_familia"].items()})
    (RAIZ / "resultados" / "sonda_t3.json").write_text(json.dumps(saida, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

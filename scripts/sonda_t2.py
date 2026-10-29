#!/usr/bin/env python3
"""Design probe of T2 (docs/decisao_avaliacao.md, section 3): what a trivial rule gets.

For the law-level task it measures how much the body length alone gives away; for the
provision-level task (slm/tarefas.py) it measures items, length, surface cues, the 13-token filter
and a TF-IDF logistic regression by grouped cross-validation on the training partition only,
as a difficulty indicator. Writes resultados/sonda_t2.json.

    python scripts/sonda_t2.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from slm.corpus import ler_documentos  # noqa: E402
from slm.dados import classificar_ementa  # noqa: E402
from slm.tarefas import INSERIDO, citacoes, corpo_lei, itens_t2, sem_sobreposicao  # noqa: E402
from slm.tokenizador import carregar_tokenizador  # noqa: E402

PISTAS = ("deste Código", "desta Lei", "nesta Lei", "regulamento", "Poder Executivo", "Lei nº", "§")
MAX_TOKENS = 254
ART1_OBJETO = re.compile(r"^Art\. 1º Esta Lei (altera|acrescenta|revoga|inclui|modifica|dá nova)", re.M | re.I)


def melhor_corte(n, y):
    """Balanced accuracy of the best single length threshold, in either direction (optimistic)."""
    m = max(((n[y == 1] <= t).mean() + (n[y == 0] > t).mean()) / 2 for t in np.unique(n))
    return float(max(m, 1 - m))


def nivel_lei(docs):
    y = np.array([classificar_ementa(d["texto"].split("\n\n")[1]) == "alteradora" for d in docs], dtype=int)
    n = []
    for d in docs:
        p = citacoes(corpo_lei(d["texto"]))
        n.append(len(p[1].encode()) if p else len(corpo_lei(d["texto"]).encode()))
    n = np.array(n)
    art1 = np.array([bool(ART1_OBJETO.search(d["texto"])) for d in docs])
    return {"art1_repete_a_alteracao": {"alteradora": float(art1[y == 1].mean()), "substantiva": float(art1[y == 0].mean())},
            "bytes_mediano_alteradora": int(np.median(n[y == 1])), "bytes_mediano_substantiva": int(np.median(n[y == 0])),
            "acuracia_balanceada_so_comprimento": melhor_corte(n, y)}


def main():
    tok = carregar_tokenizador(RAIZ / "tokenizador" / "B")
    treino_ids = [tok.encode(d["texto"]) for d in ler_documentos("B", "treino")]
    saida = {}
    for nome in ("B", "B_externo"):
        docs = ler_documentos(nome)
        its = [i for d in docs for i in itens_t2(d)]
        ids = [tok.encode(i["texto"]) for i in its]
        manter = [len(x) <= MAX_TOKENS for x in ids]
        its, ids = [i for i, m in zip(its, manter) if m], [x for x, m in zip(ids, manter) if m]
        y = np.array([i["rotulo"] for i in its])
        n = np.array([len(i["texto"].encode()) for i in its])
        aval = [k for k, i in enumerate(its) if i["particao"] != "treino"]
        ok = sem_sobreposicao([ids[k] for k in aval], treino_ids)
        r = {"leis": len(docs), "leis_com_aspas_desbalanceadas": sum(citacoes(corpo_lei(d["texto"])) is None for d in docs),
             "nivel_lei": nivel_lei(docs), "itens": len(its), "inseridos": int(y.sum()),
             "itens_acima_de_max_tokens": int(len(manter) - sum(manter)),
             "grupos": len({i["grupo"] for i in its}),
             "bytes_mediano": {"inserido": int(np.median(n[y == 1])), "proprio": int(np.median(n[y == 0]))},
             "acuracia_balanceada_so_comprimento": melhor_corte(n, y),
             "pistas": {p: {"inserido": float(np.mean([p in i["texto"] for i in its if i["rotulo"] == INSERIDO])),
                            "proprio": float(np.mean([p in i["texto"] for i in its if i["rotulo"] != INSERIDO]))} for p in PISTAS},
             "avaliacao_sem_sobreposicao_13": {}}
        for part in sorted({its[k]["particao"] for k in aval}):
            ks = [j for j, k in enumerate(aval) if its[k]["particao"] == part]
            fic = [aval[j] for j in ks if ok[j]]
            r["avaliacao_sem_sobreposicao_13"][part] = {
                "antes": len(ks), "depois": len(fic), "inseridos_depois": int(sum(its[k]["rotulo"] for k in fic)),
                "grupos_com_inserido_depois": len({its[k]["grupo"] for k in fic if its[k]["rotulo"] == INSERIDO})}
        saida[nome] = r
        print(nome, json.dumps(r, ensure_ascii=False))
    its = [i for d in ler_documentos("B", "treino") for i in itens_t2(d) if len(tok.encode(i["texto"])) <= MAX_TOKENS]
    X, y, g = [i["texto"] for i in its], np.array([i["rotulo"] for i in its]), np.array([i["grupo"] for i in its])
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(5).split(X, y, g):
        v = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2))
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(v.fit_transform([X[i] for i in tr]), y[tr])
        pred[te] = clf.predict(v.transform([X[i] for i in te]))
    saida["tfidf_cv_agrupada_treino_B"] = float(balanced_accuracy_score(y, pred))
    print("TF-IDF CV agrupada no treino:", round(saida["tfidf_cv_agrupada_treino_B"], 3))
    (RAIZ / "resultados" / "sonda_t2.json").write_text(json.dumps(saida, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

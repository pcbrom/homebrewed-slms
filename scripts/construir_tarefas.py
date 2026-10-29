#!/usr/bin/env python3
"""Build the items of the four corpus B tasks (docs/decisao_avaliacao.md, section 3).

Training items are kept whole, for the fine-tuning of the typed decider. Evaluation items
(validation, test and the external set) leave when any of their texts shares a 13-token span with
the training documents. Every item fits the 256-token window, a T2 caput with the longer of the
two prefixes the generative score puts before it. All items enter the measurement,
with the macro mean over groups as the main statistic. Writes tarefas/<corpus>/t1.jsonl to
t4.jsonl and resultados/tarefas_<corpus>.json with the counts.

    python scripts/construir_tarefas.py --corpus B --externo B_externo_atos
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from slm.corpus import ler_documentos  # noqa: E402
from slm.medida_tarefas import PREFIXO_INSERCAO  # noqa: E402
from slm.tarefas import itens_t1, itens_t2, itens_t3, itens_t4, ngramas, valores_numeros  # noqa: E402
from slm.tokenizador import carregar_tokenizador  # noqa: E402

MAX_TOKENS = 254
K = 13
PARTICOES = ("treino", "validacao", "teste", "externo")
TEXTOS = {"t1": lambda i: [i["artigo"]], "t2": lambda i: [i["texto"]],
          "t3": lambda i: [i["original"], i["adulterado"]], "t4": lambda i: [i["prefixo"] + i["continuacao"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="B")
    ap.add_argument("--externo", help="external set, default <corpus>_externo")
    a = ap.parse_args()
    tok = carregar_tokenizador(RAIZ / "tokenizador" / a.corpus)
    n_tokens = lambda t: len(tok.encode(t))
    cabe = lambda t: n_tokens(t) <= MAX_TOKENS
    treino = ler_documentos(a.corpus, "treino")
    docs = ler_documentos(a.corpus) + ler_documentos(a.externo or f"{a.corpus}_externo")
    vistos = set().union(*(ngramas(tok.encode(d["texto"]), K) for d in treino))
    limpo = lambda t: not (ngramas(tok.encode(t), K) & vistos)
    pool = valores_numeros(treino)
    tarefas = {"t1": itens_t1(docs, n_tokens, pool, cabe), "t2": [i for d in docs for i in itens_t2(d) if cabe(PREFIXO_INSERCAO + i["texto"])],
               # per partition, so that the frequency balance of T3 holds inside each evaluation set
               "t3": [i for p in PARTICOES for i in itens_t3([d for d in docs if d["particao"] == p], cabe)],
               "t4": itens_t4(docs, cabe)}
    pasta = RAIZ / "tarefas" / a.corpus
    pasta.mkdir(parents=True, exist_ok=True)
    resumo = {"max_tokens": MAX_TOKENS, "filtro_k": K, "tarefas": {}}
    for nome, its in tarefas.items():
        fic = [i for i in its if i["particao"] == "treino" or all(limpo(t) for t in TEXTOS[nome](i))]
        with open(pasta / f"{nome}.jsonl", "w", encoding="utf-8") as f:
            for i in fic:
                f.write(json.dumps(i, ensure_ascii=False) + "\n")
        chave = (lambda i: i["rotulo"]) if nome == "t2" else (lambda i: i.get("familia", ""))
        resumo["tarefas"][nome] = {
            p: {"antes": sum(i["particao"] == p for i in its), "depois": sum(i["particao"] == p for i in fic),
                "grupos": len({i["grupo"] for i in fic if i["particao"] == p}),
                "por_classe": dict(Counter(str(chave(i)) for i in fic if i["particao"] == p))}
            for p in PARTICOES}
        print(nome, json.dumps(resumo["tarefas"][nome], ensure_ascii=False))
    (RAIZ / "resultados" / f"tarefas_{a.corpus}.json").write_text(json.dumps(resumo, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

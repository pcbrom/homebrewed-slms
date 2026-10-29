#!/usr/bin/env python3
"""Surface-frequency diagnostic of the T1 and T3 items of corpus B, from the stored choices.

A scorer that ranks candidates by likelihood can prefer the candidate whose value is more frequent
in training, whatever the context. For each external item the value that tells the candidates apart
(the number or the cited article in T1, the term, number or cited article in T3) is counted in the
training documents, and each scorer is measured by how often it picks the more frequent one, per
genre of the act. In T1 the reference rates are 0.25 for a uniform choice and the share of items
whose right answer is the most frequent option. T3 shuffles keep the same words and stay out.
No model is run: the choices come from resultados/m14_B/*.jsonl.

    python scripts/diagnostico_frequencia.py
"""
import difflib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scipy import stats

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from slm.corpus import ler_documentos  # noqa: E402
from slm.tarefas import NUMERO, REMISSAO, TERMOS  # noqa: E402

PASTA = RAIZ / "resultados" / "m14_B"
PONTUADORES = {"ngrama": ["ngrama6"], "ttl-160m": ["ttl-160m"], "tucano-160m": ["tucano-160m"],
               "tucano-630m": ["tucano-630m"]}
GENEROS = {"lei": "lei", "lcp": "lei", "dec": "dec"}
VALOR = re.compile(r"\d+")


def contagens(treino):
    num, rem, termo = Counter(), Counter(), Counter()
    for d in treino:
        t = d["texto"]
        num.update(int(m.group(1)) for m in NUMERO.finditer(t))
        rem.update(int(m.group(1)) for m in REMISSAO.finditer(t))
        for p in TERMOS:
            for x in p:
                termo[x] += len(re.findall(rf"(?<!\w){re.escape(x)}(?!\w)", t))
    return num, rem, termo


def checkpoints():
    ck = json.loads((PASTA / "checkpoints.json").read_text())
    return {**PONTUADORES, **{modo: ck[modo]["run_ids"] for modo in ("autoregressivo", "difusao")}}


def ler(nome, tarefa):
    return [json.loads(l) for l in open(PASTA / f"{nome}_{tarefa}.jsonl", encoding="utf-8")]


def diferenca(a, b):
    """The words of a and of b inside the first span where the two texts differ."""
    sa, sb = a.split(), b.split()
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=sa, b=sb, autojunk=False).get_opcodes():
        if op != "equal":
            return " ".join(sa[i1:i2]), " ".join(sb[j1:j2])
    return "", ""


def freq_t3(item, num, rem, termo):
    """Training counts of the original and the adulterated value, None when the family has no value."""
    f = item["familia"]
    if f == "termo":
        par = next(p for p in TERMOS if all(re.search(rf"(?<!\w){re.escape(x)}(?!\w)", item["original"] + item["adulterado"]) for x in p))
        a = next(x for x in par if re.search(rf"(?<!\w){re.escape(x)}(?!\w)", item["original"]))
        return termo[a], termo[par[1] if a == par[0] else par[0]]
    if f in ("numero", "remissao"):
        o, d = diferenca(item["original"], item["adulterado"])
        vo, vd = VALOR.search(o), VALOR.search(d)
        if not (vo and vd):
            return None
        tab = num if f == "numero" else rem
        return tab[int(vo.group())], tab[int(vd.group())]
    return None


PROPRIO = re.compile(r"^[^.;]{0,40}\b(deste|desta)\s+Decreto")
CITADA = re.compile(r"^[^;]{0,60}?\b(?:da|do)\s+(?:Lei|Decreto-Lei|Lei Complementar)\s+n[ºo°]\s*([\d.]+)")


def alvo_remissao(item, leis_treino):
    """What a cross-reference gap of a decree points to, read from the text right after the gap:
    an article of the decree itself, of a law in the training partition of corpus B, of another law,
    or not identified (the act is named elsewhere in the sentence or not at all)."""
    d = item["direita"][:120]
    if PROPRIO.match(d):
        return "proprio"
    m = CITADA.match(d)
    if m is None:
        return "nao_identificado"
    return "lei_no_treino" if m.group(1).replace(".", "") in leis_treino else "lei_fora_do_treino"


def taxa(lista):
    return {"taxa": sum(lista) / len(lista), "itens": len(lista)} if lista else None


def main():
    treino = ler_documentos("B", "treino")
    num, rem, termo = contagens(treino)
    leis_treino = {d["doc_id"].split("-")[-1] for d in treino}
    itens_t1, itens_t3 = ([i for l in open(RAIZ / "tarefas" / "B" / f"{t}.jsonl", encoding="utf-8")
                           for i in [json.loads(l)] if i["particao"] == "externo"] for t in ("t1", "t3"))
    gen = lambda i: GENEROS[i["doc_id"].split("-")[1]]

    # T1: frequency of each option's value, the most frequent option and the most frequent distractor
    t1 = []
    for i in itens_t1:
        tab = num if i["familia"] == "numero" else rem
        f = [tab[int(VALOR.search(o).group())] for o in i["opcoes"]]
        top = max(range(len(f)), key=lambda k: f[k])
        distr = max((k for k in range(len(f)) if k != i["correta"]), key=lambda k: f[k])
        t1.append({"genero": gen(i), "familia": i["familia"], "top": top, "distr": distr,
                   "empate_top": f.count(f[top]) > 1, "correta_top": top == i["correta"],
                   "alvo": alvo_remissao(i, leis_treino) if gen(i) == "dec" and i["familia"] == "remissao" else None})
    t3 = [(gen(i), i["familia"], freq_t3(i, num, rem, termo)) for i in itens_t3]

    saida = {"descricao": __doc__.split("\n\n")[0], "t1": {}, "t3": {}}
    estratos = {"todos": lambda g: True, "lei": lambda g: g == "lei", "dec": lambda g: g == "dec"}
    for e, ok in estratos.items():
        sel = [x for x in t1 if ok(x["genero"])]
        saida["t1"].setdefault("_base", {})[e] = {"itens": len(sel), "correta_mais_frequente": taxa([x["correta_top"] for x in sel]),
                                                  "empate_no_topo": taxa([x["empate_top"] for x in sel])}
        sel3 = [x for x in t3 if ok(x[0]) and x[2] is not None and x[2][0] != x[2][1]]
        saida["t3"].setdefault("_base", {})[e] = {"itens_com_valor": len(sel3),
                                                  "original_mais_frequente": taxa([x[2][0] > x[2][1] for x in sel3])}
    for nome, corridas in checkpoints().items():
        acum1, acum3 = defaultdict(list), defaultdict(list)
        for c in corridas:
            r1, r3 = ler(c, "t1"), ler(c, "t3")
            assert [(r["doc_id"], r["familia"]) for r in r1] == [(i["doc_id"], i["familia"]) for i in itens_t1], c
            assert [(r["doc_id"], r["familia"]) for r in r3] == [(i["doc_id"], i["familia"]) for i in itens_t3], c
            for e, ok in estratos.items():
                sel = [(x, r) for x, r in zip(t1, r1) if ok(x["genero"])]
                acum1[(e, "escolhe_mais_frequente")].append(sum(r["escolha"] == x["top"] for x, r in sel) / len(sel))
                acum1[(e, "escolhe_distrator_mais_frequente")].append(sum(r["escolha"] == x["distr"] for x, r in sel) / len(sel))
                acum1[(e, "acerto")].append(sum(r["acerto"] for x, r in sel) / len(sel))
                sel3 = [(x, r) for x, r in zip(t3, r3) if ok(x[0]) and x[2] is not None and x[2][0] != x[2][1]]
                # preferring the more frequent version: a hit when the original is the more frequent one
                acum3[(e, "prefere_mais_frequente")].append(sum(r["acerto"] == (x[2][0] > x[2][1]) for x, r in sel3) / len(sel3))
                for orig_mais in (True, False):
                    s = [r["acerto"] for x, r in sel3 if (x[2][0] > x[2][1]) == orig_mais]
                    if s:
                        acum3[(e, "acerto_original_mais_frequente" if orig_mais else "acerto_adulterado_mais_frequente")].append(sum(s) / len(s))
            for x, r in zip(t1, r1):
                acum1[(f"{x['genero']}:{x['familia']}", "soma")].append(r["acerto"])
                # accuracy when the right answer is, or is not, the option most frequent in training
                acum1[("correta_mais_frequente" if x["correta_top"] else "correta_nao_mais_frequente", "soma")].append(r["acerto"])
                if x["alvo"]:
                    acum1[(f"dec_remissao_{x['alvo']}", "soma")].append(r["acerto"])
            # decrees split by group size: a group of one item weighs as much as a group of many
            for t, rs, its in (("t1", r1, itens_t1), ("t3", r3, itens_t3)):
                tam = Counter(i["grupo"] for i in its if gen(i) == "dec")
                for k, sel in (("dec_grupo_de_1", 1), ("dec_grupo_maior", 0)):
                    s = [r["acerto"] for r, i in zip(rs, its) if gen(i) == "dec" and (tam[i["grupo"]] == 1) == bool(sel)]
                    (acum1 if t == "t1" else acum3)[(k, "acerto")].append(sum(s) / len(s))
        for t, acum in (("t1", acum1), ("t3", acum3)):
            saida[t][nome] = {e: {k: sum(v) / len(v) for (ee, k), v in acum.items() if ee == e}
                              for e in (*estratos, "dec_grupo_de_1", "dec_grupo_maior")}
        saida["t1"][nome]["por_classe"] = {k: {"acerto": sum(v) / len(v), "itens": len(v) // len(corridas)}
                                           for (k, kk), v in acum1.items() if kk == "soma"}
        saida["t1"][nome]["_corridas"] = len(corridas)
    for t, its in (("t1", itens_t1), ("t3", itens_t3)):
        tam = Counter(i["grupo"] for i in its if gen(i) == "dec")
        um = [i for i in its if gen(i) == "dec" and tam[i["grupo"]] == 1]
        saida[t]["_base"]["dec_grupo_de_1"] = {"grupos": len(um), "de": len(tam), "itens_em_grupos_maiores": sum(tam.values()) - len(um),
                                               "familias": dict(Counter(i["familia"] for i in um))}
        # one-sided binomial test against chance in the single-item decree groups, on the pooled hits
        acaso = 0.25 if t == "t1" else 0.5
        for nome in checkpoints():
            x = saida[t][nome]["dec_grupo_de_1"]["acerto"]
            saida[t][nome]["dec_grupo_de_1"]["p_abaixo_do_acaso"] = float(stats.binomtest(round(x * len(um)), len(um), acaso, alternative="less").pvalue)
    saida["t1"]["_base"]["dec_remissao_alvo"] = dict(Counter(x["alvo"] for x in t1 if x["alvo"]))
    destino = RAIZ / "resultados" / "diagnostico_frequencia_B.json"
    destino.write_text(json.dumps(saida, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(saida, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Milestone M14: is the model good and useful? (docs/decisao_avaliacao.md, sections 3, 5 and 7).

    rascunho      print the pre-registration of the tasks without freezing it
    preregistrar  freeze it in preregistro/m14_tarefas_<corpus>.json, with the hash of the items
    emendar       freeze the amendment of 2026-10-28 (enlarged external set), before any new measure

    python scripts/m14.py --corpus B rascunho
    python scripts/m14.py --corpus B preregistrar
"""
import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from slm.registro import conferir_preregistro, conferir_tarefas, preregistrar, sha256_arquivo  # noqa: E402

PRE = RAIZ / "preregistro"
LAYA = {"repositorio": "convaiinnovations/laya", "revisao": "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
        "checkpoint": "multilingual/", "encoder": "mmBERT-base, 322M"}
REGUAS = ("TeenyTinyLlama 160M", "Tucano 160M", "Tucano 630M")


def desenho(corpus):
    m7 = f"m7_{corpus}"
    assert conferir_preregistro(PRE, m7), "M7 of the corpus not frozen"
    itens = {t: sha256_arquivo(RAIZ / "tarefas" / corpus / f"{t}.jsonl") for t in ("t1", "t2", "t3", "t4")}
    return {
        "corpus": corpus, "conjunto_de_medida": f"{corpus}_externo", "itens_sha256": itens,
        "construcao": "scripts/construir_tarefas.py, filtro de 13 tokens no item inteiro, todos os itens sem cota",
        "checkpoints": {
            "regra": (f"por modo, o ponto do {m7} (descritivos e confirmatorio) com menor media de bits por byte na "
                      "validacao, com todas as sementes desse ponto; a medida de cada semente entra na media"),
            "preregistro_m7": m7},
        "trilha_gerativa": {
            "ajuste": "nenhum, pontuacao por verossimilhanca",
            "piso": "n-grama Kneser-Ney do corpus, ordem escolhida na validacao (resultados/ngrama_<corpus>.json), pontuado pela mesma regra",
            "reguas_publicas": {"modelos": REGUAS, "conferencia": "logits em fp16 contra fp32 numa amostra, tokenizador por ida e volta",
                                "leitura": "intervalo, sem aprovacao nem reprovacao"},
            "T1": {"autoregressivo": "log-verossimilhanca exata do artigo preenchido com cada opcao, maior vence",
                   "difusao": "condicional exata numa passada quando a lacuna tem um token, estimador condicional do LLaDA com 128 amostras e as mesmas mascaras para as opcoes nos trechos",
                   "metrica": "acuracia, media macro por grupo", "leitura": "descritiva, com intervalo"},
            "T2": {"escore": "log p(caput | prefixo de insercao) - log p(caput | prefixo de artigo proprio)",
                   "prefixo_insercao": "passa a vigorar com a seguinte redação:\n\n“Art. 1º ",
                   "prefixo_proprio": "Art. 1º ",
                   "limiar": "um por modelo, o que maximiza a acuracia balanceada nos itens de treino, aplicado sem mudanca ao conjunto externo",
                   "difusao": "o mesmo contraste pela condicional da difusao com o caput mascarado",
                   "metrica": "acuracia balanceada; por grupo, soma dos acertos com peso 1/(2 n_classe) do conjunto"},
            "T3": {"autoregressivo": "acerta quando a log-verossimilhanca do original passa a do adulterado",
                   "difusao": "NELBO com as mesmas mascaras nas duas versoes, posicoes diferentes mascaradas",
                   "metrica": "acuracia sobre os pares, media macro por grupo"},
            "T4": {"metrica": "bits por byte da continuacao real dado o prefixo, por grupo pela soma de nats sobre a soma de bytes",
                   "difusao": "limite superior pela NELBO da continuacao com o prefixo visivel, leitura assimetrica",
                   "linhas_de_base_descritivas": ["copia da continuacao do caput de treino de maior prefixo comum"],
                   "geracao_descritiva": "chrF e ROUGE-L sobre geracao no comprimento real; juiz semantico so depois do portao do M9"},
            "confirmatorias": ["T2 autoregressivo", "T2 difusao", "T3 autoregressivo", "T3 difusao", "T4 autoregressivo", "T4 difusao"],
        },
        "trilha_decisor": {
            "ajuste": "itens de treino do corpus; T3 julga cada texto sozinho, original ou adulterado",
            "encoder": "checkpoints da regra acima, media dos estados da ultima camada, cabeca linear, modo de difusao sem mascara",
            "otimizacao": ("ajuste fino completo, AdamW, 3 epocas, lote 16, taxa escolhida em {1e-5, 3e-5, 1e-4} por validacao "
                           "cruzada agrupada em 5 dobras no treino; o mesmo protocolo para o Laya"),
            "piso": "TF-IDF de 1 e 2 gramas com regressao logistica balanceada, C escolhido na mesma validacao cruzada",
            "regua_publica": {"laya": LAYA, "leitura": "intervalo, sem aprovacao nem reprovacao",
                              "secundaria": "Laya sem ajuste, e Laya com a janela inteira dele em T3"},
            "calibracao": "temperatura por validacao cruzada agrupada no treino; ECE em 15 faixas antes e depois",
            "resposta_seletiva": "cobertura com 95% de acerto, limiar de confianca escolhido na validacao cruzada do treino",
            "metrica": "acuracia balanceada em T2 e acuracia em T3, como na trilha gerativa",
            "confirmatorias": ["T2 autoregressivo", "T2 difusao", "T3 autoregressivo", "T3 difusao"],
        },
        "estatistica": {
            "unidade": "grupo de leis", "principal": "media macro por grupo",
            "intervalo": "bootstrap selvagem por grupo, R = 10.000, 95%",
            "teste": ("por grupo, D_g = metrica do modelo - metrica do piso (em T4, bits por byte do piso - do modelo); teste "
                      "do sinal bilateral sobre os grupos com D_g diferente de zero"),
            "multiplicidade": "Holm a 0,05 dentro de cada trilha",
            "leitura": "o modelo supera o piso na tarefa quando a hipotese do sinal e rejeitada por Holm a favor dele",
            "sensibilidade": "analise refeita sem o maior grupo; correlacao intragrupo relatada como descricao",
        },
    }


def emenda(corpus):
    """Amendment approved by the author on 2026-10-28, frozen before any measure on the new items."""
    assert conferir_preregistro(PRE, f"m14_tarefas_{corpus}"), "pre-registration missing or altered"
    externo = f"{corpus}_externo_atos"
    return {
        "data": "2026-10-28", "aprovada_por": "autor", "emenda_de": f"m14_tarefas_{corpus}",
        "motivo": ("com o filtro de 13 tokens no item inteiro, o conjunto externo de leis deixava T1 com 89 itens em 23 grupos "
                   "e T3 com 167 itens em 66 grupos; o teste do sinal so teria 80% de poder com 0,72 dos grupos a favor do "
                   "modelo em T3 e 0,86 em T1 no pior caso de Holm. A emenda amplia os grupos sem afrouxar o filtro"),
        "conjunto_de_medida": externo,
        "construcao_do_conjunto": (f"scripts/construir_corpus_b.py --anos 2025 2026 --particao externo --tipos LEI LCP DEC "
                                   f"--saida corpus/{externo}: leis ordinarias, leis complementares e decretos de 2025 e 2026 "
                                   "das classes substantiva e alteradora, num so agrupamento"),
        "documentos_sha256": sha256_arquivo(RAIZ / "corpus" / externo / "documentos.jsonl"),
        "construcao_dos_itens": f"scripts/construir_tarefas.py --corpus {corpus} --externo {externo}",
        "itens_sha256": {t: sha256_arquivo(RAIZ / "tarefas" / corpus / f"{t}.jsonl") for t in ("t1", "t2", "t3", "t4")},
        "T3": "um item por familia aplicavel em cada artigo, no lugar de um operador por artigo",
        "leitura_por_genero": "lei ordinaria, lei complementar e decreto relatados a parte como descricao, sem teste",
        "grade_C_tfidf": [0.01, 0.1, 1.0, 10.0, 100.0],
        "ja_medido_antes_da_emenda": {
            "conjunto": f"{corpus}_externo (300 leis)",
            "pisos": {"ngrama6": {"T1": 0.3592, "T2": 0.4966, "T3": 0.6682, "T4_bpb": 1.435}, "tfidf": {"T2": 0.5962, "T3": 0.5321}},
            "nada_mais": "nenhum checkpoint do projeto, regua publica ou Laya foi medido em tarefa antes desta emenda"},
        "inalterado": "todo o resto do pre-registro original, inclusive filtro, checkpoints, trilhas, testes e Holm",
    }


def emenda2(corpus):
    """Second amendment, 2026-10-28: T2 items must fit the window with the generative prefix."""
    assert conferir_preregistro(PRE, f"m14_tarefas_{corpus}_emenda"), "first amendment missing or altered"
    return {
        "data": "2026-10-28", "aprovada_por": "autor", "emenda_de": f"m14_tarefas_{corpus}_emenda",
        "motivo": ("o filtro de 254 tokens de T2 media o caput sozinho, e a pontuacao gerativa poe antes dele o prefixo de "
                   "insercao, de 20 tokens; 13 dos 4.960 itens de T2 passavam da janela de 256 (4 no conjunto externo, 8 no "
                   "treino e 1 na validacao), o que a difusao recusa e o modo autorregressivo pontuaria com o contexto cortado"),
        "regra": "item de T2 entra quando o caput com o prefixo de insercao cabe em 254 tokens, para todos os pontuadores",
        "construcao_dos_itens": f"scripts/construir_tarefas.py --corpus {corpus} --externo {corpus}_externo_atos",
        "itens_sha256": {t: sha256_arquivo(RAIZ / "tarefas" / corpus / f"{t}.jsonl") for t in ("t1", "t2", "t3", "t4")},
        "ja_calculado_antes_desta_emenda": ("uma cadeia de medida interrompida pela recusa da difusao gravou os escores do "
                                            "n-grama, do TF-IDF e do modo autorregressivo nos itens da primeira emenda; "
                                            "nenhum desses escores no conjunto externo foi lido, e todos sao recalculados"),
        "inalterado": "todo o resto do pre-registro e da primeira emenda",
    }


def itens(corpus, tarefa, particoes):
    with open(RAIZ / "tarefas" / corpus / f"{tarefa}.jsonl", encoding="utf-8") as f:
        return [i for i in map(json.loads, f) if i["particao"] in particoes]


def gravar(corpus, nome, tarefa, res):
    pasta = RAIZ / "resultados" / f"m14_{corpus}"
    pasta.mkdir(parents=True, exist_ok=True)
    with open(pasta / f"{nome}_{tarefa}.jsonl", "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cmd_medir_ngrama(a):
    """Per-item scores of the n-gram floor: the external set in every task, plus the training
    items of T2, where the threshold is fitted."""
    from slm.corpus import ler_documentos, tokenizar
    from slm.medida_tarefas import pontuar
    from slm.ngrama import NGramaKN
    from slm.tokenizador import carregar_tokenizador
    conferir_tarefas(PRE, RAIZ / "tarefas" / a.corpus, a.corpus)
    tok = carregar_tokenizador(RAIZ / "tokenizador" / a.corpus)
    ordem = json.loads((RAIZ / "resultados" / f"ngrama_{a.corpus}.json").read_text(encoding="utf-8"))["ordem_escolhida"]
    ng = NGramaKN(ordem, tok.vocab_size).treinar([d["ids"] for d in tokenizar(ler_documentos(a.corpus, "treino"), tok)])
    logp = lambda seqs: [ng.logp_janelas(s, 10 ** 9) for s in seqs]
    for t in ("t1", "t2", "t3", "t4"):
        res = pontuar(t, itens(a.corpus, t, {"externo", "treino"} if t == "t2" else {"externo"}), logp, tok.encode)
        gravar(a.corpus, f"ngrama{ordem}", t, res)
        print(t, len(res), flush=True)


def checkpoints(corpus):
    """Pre-registered rule: per mode, the M7 point with the lowest mean validation bits per byte,
    with all the seeds of that point."""
    pre = json.loads((PRE / f"m7_{corpus}.json").read_text(encoding="utf-8"))
    emenda = PRE / f"m7_emenda_{corpus}.json"
    conf = json.loads(emenda.read_text(encoding="utf-8"))["sementes_confirmatorias"] if emenda.exists() else pre["sementes_confirmatorias"]
    pontos = [(ep, pre["sementes_descritivas"]) for ep in pre["pontos_descritivos"]] + [(pre["ponto_confirmatorio"], conf)]
    escolha = {}
    for modo in ("autoregressivo", "difusao"):
        medias = {}
        for ep, sem in pontos:
            av = [RAIZ / "runs" / f"m7-{corpus}-{modo}-e{ep}-s{s}" / "avaliacao.json" for s in sem]
            assert all(p.exists() for p in av), f"M7 incomplete at {modo} e{ep}"
            medias[ep] = sum(json.loads(p.read_text(encoding="utf-8"))["bpb"] for p in av) / len(av)
        ep = min(medias, key=medias.get)
        sem = dict(pontos)[ep]
        escolha[modo] = {"ponto": ep, "run_ids": [f"m7-{corpus}-{modo}-e{ep}-s{s}" for s in sem],
                         "bpb_validacao_por_ponto": medias}
    return escolha


def cmd_medir_rede(a):
    """Per-item scores of the project model, every seed of the chosen point of each mode."""
    import torch
    from slm.medida_rede import pontuador_ar, pontuar_difusao
    from slm.medida_tarefas import pontuar
    from slm.modelo import ConfigRede, Transformer
    from slm.tokenizador import carregar_tokenizador
    conferir_tarefas(PRE, RAIZ / "tarefas" / a.corpus, a.corpus)
    tok = carregar_tokenizador(RAIZ / "tokenizador" / a.corpus)
    esc = checkpoints(a.corpus)
    (RAIZ / "resultados" / f"m14_{a.corpus}").mkdir(parents=True, exist_ok=True)
    (RAIZ / "resultados" / f"m14_{a.corpus}" / "checkpoints.json").write_text(json.dumps(esc, indent=1), encoding="utf-8")
    disp = "cuda" if torch.cuda.is_available() else "cpu"
    for modo, e in esc.items():
        for rid in e["run_ids"]:
            pasta = RAIZ / "runs" / rid
            cfg = json.loads((pasta / "config.json").read_text(encoding="utf-8"))
            m = Transformer(ConfigRede(vocab_size=cfg["vocab_size"], L=cfg["L"], n_layer=cfg["n_layer"], d=cfg["d"], n_head=cfg["n_head"]))
            m.load_state_dict(torch.load(pasta / "pesos.pt", map_location="cpu"))
            m.to(disp).eval()
            for t in ("t1", "t2", "t3", "t4"):
                its = itens(a.corpus, t, {"externo", "treino"} if t == "t2" else {"externo"})
                res = (pontuar(t, its, pontuador_ar(m, cfg["L"], disp), tok.encode) if modo == "autoregressivo"
                       else pontuar_difusao(t, its, m, tok.encode, cfg["L"], K=128, dispositivo=disp))
                gravar(a.corpus, rid, t, res)
            print(rid, "ok", flush=True)


GRADE_C = (0.01, 0.1, 1.0, 10.0, 100.0)   # fixed after the freeze, before any measurement


def textos_decisor(tarefa, its):
    """(text, label, group) of the decider: T2 labels each caput, T3 judges every text alone."""
    if tarefa == "t2":
        return [(i["texto"], i["rotulo"], i["grupo"]) for i in its]
    return [(t, y, i["grupo"]) for i in its for t, y in ((i["original"], 0), (i["adulterado"], 1))]


def ids_decisor(tarefa, its):
    """doc_id of each text of textos_decisor, in the same order."""
    return [i["doc_id"] for i in its for _ in range(1 if tarefa == "t2" else 2)]


def cmd_medir_tfidf(a):
    """Decider floor: TF-IDF of 1 and 2 grams with balanced logistic regression, C by grouped
    5-fold cross-validation on the training items."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    conferir_tarefas(PRE, RAIZ / "tarefas" / a.corpus, a.corpus)
    modelo = lambda c: make_pipeline(TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)),
                                     LogisticRegression(C=c, max_iter=5000, class_weight="balanced"))
    for t in ("t2", "t3"):
        X, y, g = map(list, zip(*textos_decisor(t, itens(a.corpus, t, {"treino"}))))
        y, g = np.array(y), np.array(g)
        cv = {}
        for c in GRADE_C:
            pred = np.zeros_like(y)
            for tr, te in GroupKFold(5).split(X, y, g):
                pred[te] = modelo(c).fit([X[k] for k in tr], y[tr]).predict([X[k] for k in te])
            cv[c] = float(balanced_accuracy_score(y, pred))
        c = max(cv, key=cv.get)
        m = modelo(c).fit(X, y)
        ext = itens(a.corpus, t, {"externo"})
        Xe = textos_decisor(t, ext)
        p = m.predict_proba([x for x, _, _ in Xe])[:, 1]
        res = [{"doc_id": di, "grupo": gr, "rotulo": int(yy), "p1": float(pp), "acerto": bool((pp > 0.5) == bool(yy))}
               for (_, yy, gr), pp, di in zip(Xe, p, ids_decisor(t, ext))]
        gravar(a.corpus, "tfidf", t, res)
        print(t, "C", c, "cv", {k: round(v, 3) for k, v in cv.items()}, "externo", len(res), flush=True)


TAXAS = (1e-5, 3e-5, 1e-4)


def cmd_medir_decisor(a):
    """Decider track with the project encoders: rate by grouped 5-fold cross-validation on the
    training items, temperature and selective threshold from the out-of-fold logits at that rate,
    final fit on all training items, scored on the external set."""
    import numpy as np
    import torch
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    from slm import decisor as D
    from slm.modelo import ConfigRede, Transformer
    from slm.tokenizador import carregar_tokenizador
    conferir_tarefas(PRE, RAIZ / "tarefas" / a.corpus, a.corpus)
    tok = carregar_tokenizador(RAIZ / "tokenizador" / a.corpus)
    disp = "cuda" if torch.cuda.is_available() else "cpu"
    resumo = {}
    for modo, e in checkpoints(a.corpus).items():
        for rid in e["run_ids"]:
            pasta = RAIZ / "runs" / rid
            cfg = json.loads((pasta / "config.json").read_text(encoding="utf-8"))
            enc = Transformer(ConfigRede(vocab_size=cfg["vocab_size"], L=cfg["L"], n_layer=cfg["n_layer"], d=cfg["d"],
                                         n_head=cfg["n_head"], dropout=cfg["dropout"]))
            enc.load_state_dict(torch.load(pasta / "pesos.pt", map_location="cpu"))
            ajuste = lambda s, y, lr: D.ajustar(enc, modo == "autoregressivo", s, y, lr, cfg["L"], semente=cfg["semente"], dispositivo=disp)
            for t in ("t2", "t3"):
                tr = textos_decisor(t, itens(a.corpus, t, {"treino"}))
                ex = textos_decisor(t, itens(a.corpus, t, {"externo"}))
                s_tr, y, g = [tok.encode(x) for x, _, _ in tr], np.array([r for _, r, _ in tr]), np.array([gr for _, _, gr in tr])
                oof = {}
                for lr in TAXAS:
                    z = np.zeros((len(y), 2))
                    for i_tr, i_te in GroupKFold(5).split(s_tr, y, g):
                        dec = ajuste([s_tr[k] for k in i_tr], y[i_tr], lr)
                        z[i_te] = D.logits(dec, [s_tr[k] for k in i_te], cfg["L"], dispositivo=disp)
                    oof[lr] = z
                cv = {lr: float(balanced_accuracy_score(y, z.argmax(1))) for lr, z in oof.items()}
                lr = max(cv, key=cv.get)
                T = D.temperatura(oof[lr], y)
                p_oof = D.probabilidades(oof[lr], T)
                c = D.limiar_seletivo(p_oof.max(1), p_oof.argmax(1) == y)
                dec = ajuste(s_tr, y, lr)
                z = D.logits(dec, [tok.encode(x) for x, _, _ in ex], cfg["L"], dispositivo=disp)
                ye = np.array([r for _, r, _ in ex])
                p, p_cal = D.probabilidades(z), D.probabilidades(z, T)
                sel = p_cal.max(1) >= c
                ext = itens(a.corpus, t, {"externo"})
                gravar(a.corpus, f"decisor-{rid}", t, [{"doc_id": di, "grupo": gr, "rotulo": int(r), "p1": float(q[1]), "acerto": bool(q.argmax() == r)}
                                                       for (_, r, gr), q, di in zip(ex, p_cal, ids_decisor(t, ext))])
                resumo[f"{rid}:{t}"] = {"taxa": lr, "cv": cv, "temperatura": T, "ece_externo": D.ece(p, ye),
                                        "ece_externo_calibrado": D.ece(p_cal, ye), "limiar_seletivo": c,
                                        "cobertura_externo": float(sel.mean()),
                                        "acerto_seletivo_externo": float((p_cal.argmax(1) == ye)[sel].mean()) if sel.any() else None}
                print(rid, t, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in resumo[f"{rid}:{t}"].items() if k != "cv"}, flush=True)
    (RAIZ / "resultados" / f"m14_{a.corpus}" / "decisor_resumo.json").write_text(json.dumps(resumo, indent=1), encoding="utf-8")


REGUAS_ARQ = ("ttl-160m", "tucano-160m", "tucano-630m")


def cmd_analisar(a):
    """Both tracks from the per-item files: estimate and interval of every scorer, the
    pre-registered comparisons against the floor with Holm inside each track, the descriptive
    position of the yardsticks, the analysis without the largest group and the intra-group
    correlation of the floor."""
    import numpy as np
    from slm import medida_tarefas as mt
    pasta = RAIZ / "resultados" / f"m14_{a.corpus}"
    arq = pasta / "checkpoints.json"
    esc = (json.loads(arq.read_text(encoding="utf-8")) if arq.exists()
           else {m: {"ponto": None, "run_ids": []} for m in ("autoregressivo", "difusao")})

    def ler(nome, t):
        p = pasta / f"{nome}_{t}.jsonl"
        if not p.exists():
            return None
        with open(p, encoding="utf-8") as f:
            res = [json.loads(l) for l in f]
        return [r for r in res if r.get("particao", "externo") == "externo"], [r for r in res if r.get("particao") == "treino"]

    # genre of an act from its identifier: B-lei-N, B-lcp-N or B-dec-N
    genero = lambda r: r["doc_id"].split("-")[1]

    def resumo(t, nomes, fora=None, so=None):
        lidos = [ler(n, t) for n in nomes]
        if not lidos or any(x is None for x in lidos):
            return None
        filtra = lambda rs: [r for r in rs if r["grupo"] != fora and (so is None or genero(r) == so)]
        sementes = [filtra(e) for e, _ in lidos]
        return mt.resumo_pontuador(t, sementes, [tr for _, tr in lidos]) if all(sementes) else None

    gerativos = {"ngrama": ["ngrama6"], "autoregressivo": esc["autoregressivo"]["run_ids"], "difusao": esc["difusao"]["run_ids"],
                 **{r: [r] for r in REGUAS_ARQ}}
    decisores = {"tfidf": ["tfidf"], "autoregressivo": [f"decisor-{r}" for r in esc["autoregressivo"]["run_ids"]],
                 "difusao": [f"decisor-{r}" for r in esc["difusao"]["run_ids"]], "laya": ["laya"], "laya-sem-ajuste": ["laya-sem-ajuste"]}
    saida = {"checkpoints": {m: {"ponto": e["ponto"], "sementes": len(e["run_ids"])} for m, e in esc.items()},
             "gerativa": {}, "decisor": {}, "confirmatorias": {}}

    def trilha(nome, pontuadores, tarefas, piso, confirm):
        maior = {}
        for t in tarefas:
            base = ler(pontuadores[piso][0], t)[0]
            cont = {}
            for r in base:
                cont[r["grupo"]] = cont.get(r["grupo"], 0) + 1
            maior[t] = max(cont, key=cont.get)
            rs = {k: resumo(t, v) for k, v in pontuadores.items()}
            saida[nome][t] = {k: {"estimativa": r["estimativa"], "ic95": mt.ic_selvagem(r["estimativa"], r["residuos"]),
                                  "por_semente": r["por_semente"], "grupos": len(r["grupos"])}
                              for k, r in rs.items() if r is not None}
            saida[nome][t]["_maior_grupo"] = {"grupo": maior[t], "itens": cont[maior[t]], "de": len(base)}
            generos = sorted({genero(r) for r in base})
            for k, r in rs.items():
                if r is None:
                    continue
                if k != piso:
                    saida[nome][t][k]["contra_piso"] = mt.comparar(t, r, rs[piso])
                por = {g: resumo(t, pontuadores[k], so=g) for g in generos}
                saida[nome][t][k]["por_genero"] = {g: {"estimativa": x["estimativa"], "ic95": mt.ic_selvagem(x["estimativa"], x["residuos"]),
                                                       "grupos": len(x["grupos"])} for g, x in por.items() if x is not None}
            if piso == "ngrama":
                y = [r["acerto"] for r in base] if t in ("t1", "t3") else \
                    [r["nats"] / (np.log(2) * r["bytes"]) for r in base] if t == "t4" else None
                if y is not None:
                    saida[nome][t]["_icc_piso"] = mt.icc1(y, [r["grupo"] for r in base])
        linhas = [(t, m) for t, m in confirm if resumo(t, pontuadores[m]) is not None]
        comps = [mt.comparar(t, resumo(t, pontuadores[m]), resumo(t, pontuadores[piso])) for t, m in linhas]
        sens = [mt.comparar(t, resumo(t, pontuadores[m], maior[t]), resumo(t, pontuadores[piso], maior[t])) for t, m in linhas]
        aj = mt.holm([c["p_sinal"] for c in comps]) if comps else []
        saida["confirmatorias"][nome] = [
            {"tarefa": t, "modo": m, **c, "p_holm": p, "supera_o_piso": bool(p < 0.05 and c["diferenca"] > 0),
             "sem_o_maior_grupo": s} for (t, m), c, p, s in zip(linhas, comps, aj, sens)]
        faltam = [f"{t} {m}" for t, m in confirm if (t, m) not in linhas]
        if faltam:
            saida["confirmatorias"][nome + "_pendentes"] = faltam

    trilha("gerativa", gerativos, ("t1", "t2", "t3", "t4"), "ngrama",
           [(t, m) for t in ("t2", "t3", "t4") for m in ("autoregressivo", "difusao")])
    trilha("decisor", decisores, ("t2", "t3"), "tfidf", [(t, m) for t in ("t2", "t3") for m in ("autoregressivo", "difusao")])
    for f in ("decisor_resumo.json", "laya_resumo.json", "reguas_resumo.json"):
        if (pasta / f).exists():
            saida[f.removesuffix(".json")] = json.loads((pasta / f).read_text(encoding="utf-8"))
    (RAIZ / "resultados" / f"m14_{a.corpus}.json").write_text(json.dumps(saida, indent=1, ensure_ascii=False), encoding="utf-8")
    for nome in ("gerativa", "decisor"):
        for t, d in saida[nome].items():
            print(nome, t, {k: (round(v["estimativa"], 4), [round(x, 4) for x in v["ic95"]]) for k, v in d.items() if not k.startswith("_")})
        for c in saida["confirmatorias"][nome]:
            print(" ", c["tarefa"], c["modo"], "D", round(c["diferenca"], 4), [round(x, 4) for x in c["ic95"]],
                  f"{c['grupos_a_favor']}/{c['grupos_diferentes']}", "p_holm", round(c["p_holm"], 4), "supera", c["supera_o_piso"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="B")
    ap.add_argument("etapa", choices=["rascunho", "preregistrar", "emendar", "emendar2", "medir-ngrama", "medir-tfidf", "medir-rede", "medir-decisor",
                                      "analisar"])
    a = ap.parse_args()
    if a.etapa == "analisar":
        return cmd_analisar(a)
    if a.etapa == "emendar2":
        return print(f"m14_tarefas_{a.corpus}_emenda2", preregistrar(PRE, f"m14_tarefas_{a.corpus}_emenda2", emenda2(a.corpus)))
    if a.etapa == "emendar":
        return print(f"m14_tarefas_{a.corpus}_emenda", preregistrar(PRE, f"m14_tarefas_{a.corpus}_emenda", emenda(a.corpus)))
    if a.etapa == "medir-decisor":
        return cmd_medir_decisor(a)
    if a.etapa == "medir-ngrama":
        return cmd_medir_ngrama(a)
    if a.etapa == "medir-tfidf":
        return cmd_medir_tfidf(a)
    if a.etapa == "medir-rede":
        return cmd_medir_rede(a)
    d = desenho(a.corpus)
    if a.etapa == "rascunho":
        print(json.dumps(d, indent=2, ensure_ascii=False))
    else:
        print(f"m14_tarefas_{a.corpus}", preregistrar(PRE, f"m14_tarefas_{a.corpus}", d))


if __name__ == "__main__":
    main()

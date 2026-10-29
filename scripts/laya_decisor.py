#!/usr/bin/env python3
"""Public yardstick of the decider track (preregistro/m14_tarefas_B.json): the multilingual Laya
checkpoint fine-tuned with the protocol of the project decider, and read without fine-tuning.

Same protocol as scripts/m14.py medir-decisor: full fine-tuning (the action head stays frozen, as
in the author's recipe, because it is not part of the choice), AdamW at the PyTorch defaults, one
constant rate for the whole model chosen in {1e-5, 3e-5, 1e-4} by grouped 5-fold cross-validation
on the training items, 3 epochs, batch 16, cross-entropy weighted by the inverse class frequency;
temperature and selective threshold from the out-of-fold logits. Laya runs in bf16 autocast, its
own runtime precision. Every item fits the Laya window, so no item is cut.

Runs in the environment of the Laya runtime, which has the package installed:
    <modelos_decisao>/.venv/bin/python scripts/laya_decisor.py --corpus B --laya <modelos_decisao>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from slm import decisor as D  # noqa: E402
from slm.registro import conferir_tarefas  # noqa: E402

TAXAS = (1e-5, 3e-5, 1e-4)
PERGUNTAS = {
    "t2": ("dispositivo", {"type": "choice",
                           "instructions": "Este dispositivo legal é um artigo próprio da lei que o contém ou foi escrito para ser inserido em outra norma, alterando-a?",
                           "criteria": {"proprio": "artigo próprio da lei", "inserido": "dispositivo inserido em outra norma"}}),
    "t3": ("artigo", {"type": "choice",
                      "instructions": "Este artigo de lei está no texto original ou foi adulterado, com um termo, número, remissão ou a ordem de dispositivos trocados?",
                      "criteria": {"original": "texto original", "adulterado": "texto adulterado"}}),
}


def ler(corpus, t, particao):
    with open(RAIZ / "tarefas" / corpus / f"{t}.jsonl", encoding="utf-8") as f:
        its = [i for i in map(json.loads, f) if i["particao"] == particao]
    if t == "t2":
        return [(i["texto"], i["rotulo"], i["grupo"], i["doc_id"]) for i in its]
    return [(x, y, i["grupo"], i["doc_id"]) for i in its for x, y in ((i["original"], 0), (i["adulterado"], 1))]


def codificar(agent, t, textos):
    chave, q = PERGUNTAS[t]
    internal = {"q": agent._to_internal(q)}
    assert list(internal["q"]["crit"]) == list(q["criteria"]), "option order changed"
    return [agent._encode_state({chave: x}, ["q"], internal)[0] for x in textos]


def passo(agent, its, collate):
    b = collate([[it] for it in its], agent.tok.pad_token_id)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z, _ = agent.model(b["input_ids"].cuda(), b["attention_mask"].cuda(), b["marker_pos"].cuda(),
                           b["marker_mask"].cuda(), b["qtype"].cuda())
    return z.float().masked_fill(~b["marker_mask"].cuda(), -1e4)[:, :2]


def ajustar(agent, base, its, y, lr, semente, collate):
    agent.model.load_state_dict(base)
    for p in agent.model.act_head.parameters():
        p.requires_grad_(False)
    params = [p for p in agent.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    cont = np.bincount(y, minlength=2)
    peso = torch.tensor(len(y) / (2 * cont), dtype=torch.float32, device="cuda")
    torch.manual_seed(semente)
    rng = np.random.default_rng(semente)
    agent.model.train()
    for _ in range(3):
        for b in np.array_split(rng.permutation(len(y)), max(1, -(-len(y) // 16))):
            perda = F.cross_entropy(passo(agent, [its[k] for k in b], collate), torch.as_tensor(y[b], device="cuda"), weight=peso)
            opt.zero_grad(set_to_none=True)
            perda.backward()
            opt.step()
    agent.model.eval()


@torch.no_grad()
def logits(agent, its, collate, lote=64):
    agent.model.eval()
    return np.concatenate([passo(agent, its[k:k + lote], collate).cpu().numpy() for k in range(0, len(its), lote)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="B")
    ap.add_argument("--laya", required=True, help="folder of the Laya runtime (laya-pesos/ inside)")
    ap.add_argument("--semente", type=int, default=13)
    a = ap.parse_args()
    conferir_tarefas(RAIZ / "preregistro", RAIZ / "tarefas" / a.corpus, a.corpus)
    import laya
    from laya.common import collate_items
    pre = json.loads((RAIZ / "preregistro" / f"m14_tarefas_{a.corpus}.json").read_text(encoding="utf-8"))
    rev = (Path(a.laya) / "laya-pesos" / "REVISION").read_text().strip()
    assert rev == pre["trilha_decisor"]["regua_publica"]["laya"]["revisao"], "Laya revision differs from the pre-registered one"
    agent = laya.load(str(Path(a.laya) / "laya-pesos" / "multilingual"), device="cuda")
    base = {k: v.detach().cpu().clone() for k, v in agent.model.state_dict().items()}
    saida = RAIZ / "resultados" / f"m14_{a.corpus}"
    resumo = {"revisao": rev, "semente": a.semente}
    for t in ("t2", "t3"):
        tr, ex = ler(a.corpus, t, "treino"), ler(a.corpus, t, "externo")
        i_tr, i_ex = codificar(agent, t, [x for x, _, _, _ in tr]), codificar(agent, t, [x for x, _, _, _ in ex])
        y, g = np.array([r for _, r, _, _ in tr]), np.array([gr for _, _, gr, _ in tr])
        ye = np.array([r for _, r, _, _ in ex])
        agent.model.load_state_dict(base)
        z0 = logits(agent, i_ex, collate_items)
        oof = {}
        for lr in TAXAS:
            z = np.zeros((len(y), 2))
            for k_tr, k_te in GroupKFold(5).split(i_tr, y, g):
                ajustar(agent, base, [i_tr[k] for k in k_tr], y[k_tr], lr, a.semente, collate_items)
                z[k_te] = logits(agent, [i_tr[k] for k in k_te], collate_items)
            oof[lr] = z
        cv = {lr: float(balanced_accuracy_score(y, z.argmax(1))) for lr, z in oof.items()}
        lr = max(cv, key=cv.get)
        T = D.temperatura(oof[lr], y)
        p_oof = D.probabilidades(oof[lr], T)
        c = D.limiar_seletivo(p_oof.max(1), p_oof.argmax(1) == y)
        ajustar(agent, base, i_tr, y, lr, a.semente, collate_items)
        z = logits(agent, i_ex, collate_items)
        p, p_cal, p0 = D.probabilidades(z), D.probabilidades(z, T), D.probabilidades(z0)
        sel = p_cal.max(1) >= c
        for nome, pp in (("laya", p_cal), ("laya-sem-ajuste", p0)):
            with open(saida / f"{nome}_{t}.jsonl", "w", encoding="utf-8") as f:
                for (_, r, gr, di), q in zip(ex, pp):
                    f.write(json.dumps({"doc_id": di, "grupo": gr, "rotulo": int(r), "p1": float(q[1]), "acerto": bool(q.argmax() == r)}) + "\n")
        resumo[t] = {"taxa": lr, "cv": cv, "temperatura": T, "ece_externo": D.ece(p, ye), "ece_externo_calibrado": D.ece(p_cal, ye),
                     "limiar_seletivo": c, "cobertura_externo": float(sel.mean()),
                     "acerto_seletivo_externo": float((p_cal.argmax(1) == ye)[sel].mean()) if sel.any() else None,
                     "acuracia_balanceada_sem_ajuste": float(balanced_accuracy_score(ye, p0.argmax(1)))}
        print(t, json.dumps(resumo[t]), flush=True)
    (saida / "laya_resumo.json").write_text(json.dumps(resumo, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

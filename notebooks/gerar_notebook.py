#!/usr/bin/env python3
"""Build the two class notebooks, one per generation mode, one section per box of the pipeline.

notebooks/slm_autorregressivo.ipynb and notebooks/slm_difusao.ipynb share the preparation boxes
and differ in the branch of boxes 4, 6 and 7, in the building blocks and in the technique notes.
Needs Pillow (a matplotlib dependency) to embed the pipeline diagram as a cell attachment, so it
renders offline and before the repository is public. The notebooks clone the repository in Colab,
or use the local checkout when the environment variable SLM_LOCAL points to it. Every box has
three parts: theory (building blocks from notebooks/teoria.py), application (the code below) and
technique (problems and solutions from notebooks/tecnica.py).

    python notebooks/gerar_notebook.py
"""
import base64
import io
import json
import re
import sys
from pathlib import Path

from PIL import Image

PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA))
from tecnica import TECNICA  # noqa: E402
from teoria import TEORIA, mapa  # noqa: E402

REPO = "https://github.com/pcbrom/homebrewed-slms"
ARQUIVOS = {"autoregressivo": "slm_autorregressivo.ipynb", "difusao": "slm_difusao.ipynb"}
NOME = {"autoregressivo": "autorregressivo", "difusao": "difusão"}


def anexo(nome: str, largura: int = 900) -> dict:
    img = Image.open(PASTA.parent / "figuras" / nome)
    img = img.resize((largura, round(img.height * largura / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return {nome: {"image/png": base64.b64encode(buf.getvalue()).decode()}}


def md(texto: str, anexos: dict | None = None) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": texto.strip("\n"),
            **({"attachments": anexos} if anexos else {})}


def code(texto: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": texto.strip("\n")}


def celulas(modo: str) -> list[dict]:
    ar = modo == "autoregressivo"
    outro = ARQUIVOS["difusao" if ar else "autoregressivo"]
    c = []
    if ar:
        c.append(md(r"""
# Modelo de linguagem pequeno do zero, modo autorregressivo

Este notebook constrói do zero um modelo de linguagem pequeno que gera texto da esquerda para a direita, um token por vez, como os grandes modelos de linguagem atuais. O notebook irmão, `{outro}`, treina a mesma arquitetura, com pesos próprios, no modo de difusão mascarada, e os dois repetem de propósito as caixas que têm em comum, para que cada um possa ser lido sozinho.
"""))
    else:
        c.append(md(r"""
# Modelo de linguagem pequeno do zero, modo de difusão mascarada

Este notebook constrói do zero um modelo de linguagem pequeno que gera texto partindo de uma janela toda mascarada e revelando posições em ordem aleatória, até completar o texto. O notebook irmão, `{outro}`, treina a mesma arquitetura, com pesos próprios, no modo autorregressivo, e os dois repetem de propósito as caixas que têm em comum, para que cada um possa ser lido sozinho.
"""))
    c[-1]["source"] = c[-1]["source"].replace("{outro}", outro) + "\n\n" + r"""
Cada seção corresponde a uma caixa do diagrama de arquitetura do projeto e é contada em três partes. A teoria vem decomposta em blocos de construção matemáticos, a aplicação roda o código e mostra o que entra e o que sai, e a técnica traz os problemas que o projeto encontrou e como foram resolvidos. Nada aqui usa pesos ou tokenizador pré-treinados.

Projeto, decisões e referências em [{REPO}]({REPO}). Para rodar no Colab, escolha **Ambiente de execução, Alterar o tipo de ambiente de execução, GPU T4** e execute as células em ordem.

Se este material for útil para você, deixe uma estrela no repositório e cite-o pelo `CITATION.cff`.

O pacote `slm/` foi escrito com identificadores em português, como `treinar`, `janelar`, `mascarar` e `perda`, então o código se lê na mesma língua do texto. Por convenção de programação, os identificadores não levam acento, e por isso o modo aparece no código como `autoregressivo` ou `difusao`.
""".replace("{REPO}", REPO).strip("\n")

    c.append(code(r"""
import os, subprocess, sys
LOCAL = os.environ.get("SLM_LOCAL")
if LOCAL:
    RAIZ = LOCAL
else:
    RAIZ = "/content/homebrewed-slms"
    if not os.path.exists(RAIZ):
        subprocess.run(["git", "clone", "--depth", "1", "{REPO}", RAIZ], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tokenizers==0.23.3", "regex"], check=True)
sys.path.insert(0, RAIZ)
os.chdir(RAIZ)
import torch
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
print("dispositivo:", DISPOSITIVO, torch.cuda.get_device_name(0) if DISPOSITIVO == "cuda" else "")
""".replace("{REPO}", REPO)))

    ramo = ("Este notebook segue o ramo autorregressivo da caixa 4, com máscara de atenção causal."
            if ar else "Este notebook segue o ramo de difusão da caixa 4, em que tokens são trocados por `[MASK]`.")
    c.append(md(r"""
## O fluxo completo

![fluxo](attachment:pipeline.png)

A preparação (caixas 0 a 2) roda uma única vez e deixa no repositório o corpus limpo e o tokenizador. O treino (caixas 3 a 6) é igual nos dois modos até a caixa 4, onde a difusão troca tokens por `[MASK]` e o modo autorregressivo usa atenção causal. A avaliação e a geração (caixas 7 a 10) medem o que o modelo aprendeu. """ + ramo, anexo("pipeline.png")))
    c.append(md(mapa(modo)))

    c.append(md(r"""
## Caixa 0. Configuração

Uma corrida é descrita por um único objeto. Trocar o domínio, o tamanho da rede ou o modo significa mudar campos aqui, e nada mais. `CORPUS` escolhe o texto de treino, `"A"` para a prosa de Machado de Assis e `"B"` para leis federais brasileiras, e `EPOCAS_AULA` define quantas vezes o treino ao vivo percorre o texto.

Antes de treinar, vale estimar quanto texto a sua GPU processa por hora e quanto texto o modelo precisa ver. A dedução completa está em `docs/DIMENSIONAMENTO.md`, e a célula abaixo roda o mesmo cálculo para uma T4.
"""))
    c.append(code(r"""
from slm.config import Config
MODO = "{MODO}"
CORPUS = os.environ.get("SLM_CORPUS", "A")
EPOCAS_AULA = float(os.environ.get("SLM_EPOCAS", "16"))
cfg_base = dict(L=256, n_layer=6, d=384, n_head=6, lote=64, epocas=EPOCAS_AULA, lr=1e-3,
                precisao="fp16" if DISPOSITIVO == "cuda" else "fp32",
                atencao="efficient" if DISPOSITIVO == "cuda" else "math", papel="aula", corpus=CORPUS)
print("modo:", MODO, "|", cfg_base)
print(subprocess.run([sys.executable, "scripts/dimensionar.py", "--pico-tflops", "65", "--vram-gb", "16",
                      "--arq", "6x384", "--vocab", "4096", "--horas", "0.25"],
                     check=True, capture_output=True, text=True).stdout)
""".replace("{MODO}", modo)))

    c.append(md(r"""
## Caixa 1. Adaptador de domínio

Entra o texto de uma fonte e saem documentos limpos em NFC, cada um com grupo, partição e contagem de bytes.

O corpus A reúne as obras em prosa de Machado de Assis do Projeto Gutenberg, em domínio público e na grafia original das edições de época ("elle", "annos"). Cada capítulo é um documento. A divisão reserva para validação e teste blocos contíguos de capítulos inteiros de cada obra, com um capítulo de margem antes de cada bloco retido, e o tokenizador só é treinado depois da divisão, para que nenhuma estatística do texto de avaliação vaze para o vocabulário.
"""))
    c.append(code(r"""
import collections
from slm.corpus import ler_documentos
docs = ler_documentos(CORPUS)
por_particao = collections.Counter(d["particao"] for d in docs)
bytes_particao = collections.Counter()
for d in docs:
    bytes_particao[d["particao"]] += d["bytes"]
for p in ("treino", "margem", "validacao", "teste"):
    print(f"{p:10s} {por_particao[p]:5d} documentos {bytes_particao[p] / 1e6:7.3f} MB")
exemplo = next(d for d in docs if d["particao"] == "treino")
print("\nexemplo:", exemplo["doc_id"], exemplo["grupo"], exemplo["bytes"], "bytes")
print(exemplo["texto"][:600])
"""))

    c.append(md(r"""
## Caixa 2. Tokenizador

Entra texto e sai uma sequência de ids inteiros, com o caminho de volta.

O BPE em nível de byte parte dos 256 bytes como vocabulário e, a cada iteração, funde o par adjacente mais frequente num símbolo novo. Como qualquer texto é uma sequência de bytes, não existe token desconhecido, e $\mathrm{decode}(\mathrm{encode}(s)) = s$ para toda cadeia. Os ids de 0 a 3 ficam reservados para `[PAD]`, `[BOS]`, `[EOS]` e `[MASK]` e nunca são produzidos a partir de texto.

Abaixo, a implementação didática treina numa amostra pequena e mostra as primeiras fusões, como texto quando os bytes formam caracteres inteiros e como bytes crus quando não formam. Em seguida, o tokenizador do experimento, treinado pela biblioteca sobre o corpus inteiro, codifica uma frase. As duas implementações produzem as mesmas fusões e os mesmos ids, o que os testes do projeto verificam. O notebook do tokenizador, `slm_tokenizador.ipynb`, aprofunda esta caixa.
"""))
    c.append(code(r"""
from slm.tokenizador import BPEDidatico, BPEBiblioteca
amostra = [d["texto"] for d in docs if d["particao"] == "treino"][:20]
didatico = BPEDidatico().treinar(amostra, 256 + 4 + 40)
def mostrar(x):
    try:
        return repr(x.decode("utf-8"))
    except UnicodeDecodeError:
        return repr(x)
for a, b in didatico.merges[:15]:
    print(f"{mostrar(a):>12} + {mostrar(b):<12} -> {mostrar(a + b)}")
tok = BPEBiblioteca.carregar(f"tokenizador/{CORPUS}")
frase = "Capitú tinha olhos de ressaca, disse-lhe elle n'aquella hora."
ids = tok.encode(frase)
print("\nV =", tok.vocab_size, "| ids:", ids)
print("pedaços:", [tok.decode([i]).decode("utf-8", "replace") for i in ids])
print("ida e volta exata:", tok.decode(ids) == frase.encode("utf-8"))
"""))

    c.append(md(r"""
## Caixa 3. Janelamento

Entram os ids de um documento e saem janelas de $L$ tokens.

Cada documento recebe um `[EOS]` no fim e é cortado em $\lceil (n+1)/L \rceil$ janelas de $L$ tokens, a última completada com `[EOS]`. O preenchimento é previsto e cobrado nos dois modos. Se a difusão recebesse o comprimento de graça, ganharia até $\log_2 L$ bits por documento sobre o modo autorregressivo, que paga pelo seu `[EOS]`.
"""))
    c.append(code(r"""
from slm.corpus import tokenizar
from slm.janelas import janelar, janelas_documento, fracao_preenchimento
treino = tokenizar(ler_documentos(CORPUS, "treino"), tok)
validacao = tokenizar(ler_documentos(CORPUS, "validacao"), tok)
L = cfg_base["L"]
J = janelas_documento(treino[0]["ids"], L)
print("um documento com", len(treino[0]["ids"]), "tokens vira", tuple(J.shape), "janelas")
print("fim da última janela:", J[-1, -12:].tolist())
janelas_treino, _ = janelar([d["ids"] for d in treino], L)
print("janelas de treino:", tuple(janelas_treino.shape), "| fração de preenchimento:",
      round(fracao_preenchimento([d["ids"] for d in treino], L), 4))
"""))

    if ar:
        c.append(md(r"""
## Caixa 4. Entrada deslocada e máscara causal

O texto fica intacto. A entrada é a janela deslocada uma posição para a direita, com um `[BOS]` no início, e a máscara de atenção causal impede cada posição de ver as seguintes. Assim, a saída na posição $i$ prevê o token $w_i$ olhando só para os tokens anteriores a ele.
"""))
        c.append(code(r"""
from slm.perda import entrada_ar
def texto_de(ids):
    return "".join("⟨BOS⟩" if i == 1 else tok.decode([i]).decode("utf-8", "replace") for i in ids)
w = janelas_treino[:1, :16]
print("alvo    |", texto_de(w[0].tolist()))
print("entrada |", texto_de(entrada_ar(w)[0].tolist()))
causal = torch.tril(torch.ones(6, 6, dtype=torch.int))
print("\nmáscara causal 6x6 (1 = pode olhar):\n", causal)
"""))
    else:
        c.append(md(r"""
## Caixa 4. Mascaramento

Sorteia-se $l \sim U\{1, \dots, L\}$, estratificado no lote, e exatamente $l$ posições escolhidas de modo uniforme viram `[MASK]`. A rede vê o texto com lacunas e tenta preenchê-las. Abaixo, cada linha é o começo de uma janela com o seu próprio $l$, e o símbolo ▒ marca as posições mascaradas.
"""))
        c.append(code(r"""
from slm.mascaramento import sortear_l, mascara_de_l, mascarar
g = torch.Generator().manual_seed(0)
w = janelas_treino[:4, :24]
l = sortear_l(4, 24, g)
z = mascarar(w, mascara_de_l(l, 24, g))
def texto_de(ids):
    return "".join("▒" if i == 3 else tok.decode([i]).decode("utf-8", "replace") for i in ids)
for i in range(4):
    print(f"l = {int(l[i]):2d} |", texto_de(z[i].tolist()))
"""))

    c.append(md(r"""
## Caixa 5. Rede Transformer

Entram ids, $(B, L)$, e saem logits, $(B, L, V)$.

A mesma arquitetura serve aos dois modos e não recebe entrada de tempo. Na difusão, Ou et al. (2024) mostraram que a distribuição que a rede precisa aprender, a dos tokens limpos dado o contexto visível, não depende do tempo, como detalha a técnica desta caixa. Os blocos aplicam LayerNorm antes da atenção e antes do MLP, as posições são aprendidas, e o embedding é compartilhado entre entrada e saída. O logit de `[MASK]` é fixado em $-\infty$, e a rede nunca prevê a máscara. """
        + ("Neste notebook a rede roda sempre com a máscara causal." if ar else "Neste notebook a rede roda sem máscara causal, e cada posição vê a janela inteira.")))
    c.append(code(r"""
from slm.modelo import ConfigRede, Transformer
rede = Transformer(ConfigRede(vocab_size=tok.vocab_size, L=L, n_layer=cfg_base["n_layer"], d=cfg_base["d"],
                              n_head=cfg_base["n_head"]))
print(rede.n_parametros())
with torch.no_grad():
    saida = rede(janelas_treino[:2], causal=MODO == "autoregressivo")
print("entrada", tuple(janelas_treino[:2].shape), "-> logits", tuple(saida.shape))
print("logit de [MASK]:", float(saida[0, 0, 3]))
"""))

    if ar:
        perda = r"""
$$\ell_{AR}(w) = \sum_{i=1}^{L} -\log p_\theta(w_i \mid \text{BOS}, w_{<i}),$$

a log-verossimilhança negativa exata da janela. É ela que a avaliação da caixa 8 converte em bits por byte."""
    else:
        perda = r"""
Com $l$ máscaras no conjunto $S$,

$$\hat\ell_{dif}(w; l, S) = \frac{L}{l} \sum_{i \in S} -\log p_\theta(w_i \mid z), \qquad \mathbb{E}_{l, S}\big[\hat\ell_{dif}\big] = \mathrm{NELBO}(w) \ge -\log p_\theta(w).$$

A esperança é a NELBO exata da difusão com cronograma linear, sem corte em $t$ pequeno, porque $\int_0^1 \binom{L}{l} t^{l-1}(1-t)^{L-l}\,dt = 1/l$."""
    c.append(md(r"""
## Caixa 6. Perda e otimização
""" + perda + r"""

A perda é calculada em fp32 a partir dos logits, o treino roda em fp16 com escala de gradiente na T4, e só os passos efetivamente aplicados contam para o orçamento. O treino abaixo é curto, dimensionado para uma aula. O notebook irmão treina o outro modo com a mesma arquitetura e a mesma inicialização, os mesmos dados, a mesma ordem e o mesmo número de FLOPs, o que torna as duas corridas comparáveis.
"""))
    c.append(code(r"""
from slm.treino import treinar
cfg = Config(modo=MODO, vocab_size=tok.vocab_size, **cfg_base).validar()
torch.manual_seed(cfg.semente)
modelo = Transformer(ConfigRede(vocab_size=cfg.vocab_size, L=cfg.L, n_layer=cfg.n_layer, d=cfg.d,
                                n_head=cfg.n_head)).to(DISPOSITIVO)
resumo = treinar(cfg, modelo, janelas_treino, DISPOSITIVO)
modelo.eval()
print({k: round(v, 3) if isinstance(v, float) else v for k, v in resumo.items()})
"""))

    if ar:
        c.append(md(r"""
## Caixa 7. Geração

Um token por vez, da esquerda para a direita. A partir do `[BOS]`, a rede dá a distribuição do próximo token, sorteia-se um, ele entra na entrada e o passo se repete até o `[EOS]` ou o fim da janela. O sorteio é feito em float64, porque a truncagem do ruído do sorteio já reduz a diversidade das amostras em fp32 (Zheng et al., 2024), e em fp16 o efeito seria maior.
"""))
        c.append(code(r"""
from slm.geracao import gerar_ar, ate_eos
def como_texto(ids):
    return tok.decode(ate_eos([i for i in ids if i > 3 or i == 2])).decode("utf-8", "replace")
amostras = gerar_ar(modelo, L, n=3, semente=1, dispositivo=DISPOSITIVO)
for a in amostras.tolist():
    print("AR |", como_texto(a), "\n")
"""))
    else:
        c.append(md(r"""
## Caixa 7. Geração

A janela começa toda mascarada e, a cada passo de $t = k/T$ para $s = (k-1)/T$, cada posição ainda mascarada é revelada com probabilidade $(t-s)/t$. Com $T$ passos, o texto inteiro aparece em paralelo, e não da esquerda para a direita. A janela tem as mesmas $L$ posições do treino, porque a rede bidirecional só viu janelas desse tamanho, e o texto é cortado no primeiro `[EOS]`. O sorteio é feito em float64, porque a truncagem do ruído do sorteio já reduz a diversidade das amostras em fp32 (Zheng et al., 2024), e em fp16 o efeito seria maior.
"""))
        c.append(code(r"""
from slm.geracao import gerar_difusao, ate_eos
def como_texto(ids):
    return tok.decode(ate_eos([i for i in ids if i > 3 or i == 2])).decode("utf-8", "replace")
amostras = gerar_difusao(modelo, L, T_gen=L, n=3, semente=1, dispositivo=DISPOSITIVO)
for a in amostras.tolist():
    print("DIF |", como_texto(a), "\n")
"""))

    leitura = ("No modo autorregressivo a NLL é exata, e o número obtido aqui é a log-verossimilhança verdadeira do modelo. É ele que serve de régua quando os dois modos são comparados."
               if ar else "Na difusão a NLL é a NELBO, um limite superior. Por isso só uma vitória da difusão sobre o modo autorregressivo é conclusiva, já que, se ela perder, a diferença pode ser folga do limite.")
    c.append(md(r"""
## Caixa 8. Avaliação em bits por byte

$$\mathrm{bpb} = \frac{\sum_d \mathrm{NLL}(d)}{\ln 2 \cdot \sum_d B_d}$$

Medir por byte, e não por token, põe na mesma escala modelos com tokenizadores diferentes, e por isso os bits por byte servem para comparar configurações diferentes. """ + leitura))
    c.append(code(r"""
from slm.avaliacao import avaliar
res = avaliar(modelo, validacao, MODO, L, K=1, dispositivo=DISPOSITIVO)
print(f"{MODO} | bpb de validação = {res['bpb']:.4f}")
"""))
    c.append(md(r"""
### Confiança declarada contra acurácia observada

Num modelo bem calibrado, as previsões feitas com confiança declarada de 0,7 acertam em cerca de 70% dos casos. A temperatura $T$ divide os logits antes do softmax e é ajustada pelo método de Newton na validação. Pontos abaixo da diagonal indicam excesso de confiança, e pontos acima, falta de confiança.

Dois cuidados entram antes da medida. As janelas da validação vão em ordem sorteada, porque o limite de previsões coletadas, na ordem original, ficaria só com as primeiras obras. E o preenchimento depois do `[EOS]` fica de fora: ele é previsto quase com certeza e, no projeto, chegou a ocupar a maior parte das previsões de alta confiança, o que fazia a calibração medir o preenchimento e não o texto.
"""))
    c.append(code(r"""
import torch
from slm import calibracao as cal, graficos
janelas_val, _ = janelar([d["ids"] for d in validacao], L)
janelas_val = janelas_val[torch.randperm(janelas_val.shape[0], generator=torch.Generator().manual_seed(0))]
Z, y = cal.coletar_previsoes(modelo, janelas_val, MODO, n_max=20000, dispositivo=DISPOSITIVO, sem_preenchimento=True)
T = cal.temperatura_newton(Z, y)
sistemas = {}
for rotulo, t in ((f"{MODO} T=1", 1.0), (f"{MODO} calibrado", T)):
    cf, ac = cal.confianca_acerto(Z, y, t)
    sistemas[rotulo] = dict(cal.confiabilidade(cf, ac), ece=cal.ece(cf, ac))
print(MODO, "| T =", round(T, 3), "| AUC ROC =", round(cal.auc_roc(*cal.confianca_acerto(Z, y, T)), 3))
fig = graficos.confiabilidade(sistemas)
"""))

    c.append(md(r"""
## Caixa 9. Incerteza

Um número de bits por byte só vira afirmação acompanhado da sua incerteza. O protocolo do projeto usa várias sementes por modo e um bootstrap que reamostra obras e sementes. Aqui, com uma única semente, o intervalo abaixo reamostra só as obras, e por isso mede a variação entre partes do texto, não entre corridas de treino. A comparação entre os dois modos usa a diferença pareada por semente, descrita nos blocos desta caixa, e os resultados do protocolo completo estão em `docs/`, no repositório.
"""))
    c.append(code(r"""
import numpy as np
grupos = sorted({r["grupo"] for r in res["documentos"]})
por_grupo = {g: [(r["nats"], r["bytes"]) for r in res["documentos"] if r["grupo"] == g] for g in grupos}
rng = np.random.default_rng(0)
replicas = []
for _ in range(2000):
    escolha = rng.choice(grupos, size=len(grupos), replace=True)
    nats = sum(n for g in escolha for n, _ in por_grupo[g])
    nbytes = sum(b for g in escolha for _, b in por_grupo[g])
    replicas.append(nats / nbytes / np.log(2))
print(f"{MODO} | bpb = {res['bpb']:.4f} | intervalo de 95% entre obras:", np.percentile(replicas, [2.5, 97.5]).round(4))
"""))

    c.append(md(r"""
## Caixa 10. Leitura e memorização

Um modelo treinado por muitas épocas num corpus pequeno pode parecer coerente porque copia o texto de treino. Antes de qualquer leitura humana, cada amostra passa por dois filtros automáticos, a fração das suas palavras presentes no léxico de treino (nível N1) e o comprimento do maior trecho copiado do texto de treino.
"""))
    c.append(code(r"""
from slm.leitura import IndiceTreino, lexico, taxa_lexico
textos_treino = [d["texto"] for d in ler_documentos(CORPUS, "treino")]
lex = lexico(textos_treino)
indice = IndiceTreino(textos_treino)
for a in amostras.tolist():
    t = como_texto(a)
    print("taxa de léxico:", round(taxa_lexico(t, lex), 3), "| maior trecho copiado:", indice.maior_trecho(t), "palavras")
"""))
    c.append(md(r"""
### Curva de deleção

Quanto uma previsão depende do contexto? Para cada trecho de 50 palavras, palavras do contexto são removidas, começando pelas que mais sustentam a decisão, e a probabilidade do token decidido é acompanhada em relação à original. Com 100% removido o contexto fica vazio, e o que resta é o viés intrínseco do modelo. A curva com ordem aleatória serve de referência.
"""))
    c.append(code(r"""
from slm import delecao
itens = delecao.itens([d["texto"] for d in ler_documentos(CORPUS, "teste")], tok, n_itens=20, n_palavras=50)
res_del = delecao.analisar(modelo, MODO, tok, itens, L, T=T, dispositivo=DISPOSITIVO)
print(f"AUC por importância {res_del['auc_importancia_media']:.3f} | aleatória {res_del['auc_aleatoria_media']:.3f} | "
      f"contexto vazio {res_del['p_vazio_relativo_medio']:.3f} | Spearman p0 x AUC {res_del['spearman_p0_auc']['rho']:.2f}")
fig = graficos.delecao({MODO: res_del})
"""))

    if ar:
        exercicios = r"""
1. Na caixa 0, defina `CORPUS = "B"` e rode tudo de novo. O que muda no código e o que continua igual?
2. Na caixa 7, passe a `gerar_ar` um `prefixo` com os ids de uma frase sua. Até onde a continuação mantém o assunto?
3. Na caixa 0, dobre `EPOCAS_AULA`. Os bits por byte de validação caem na mesma proporção que a perda de treino? O que a diferença indica?
4. Na caixa 4, o que aconteceria com a perda de treino se a entrada não fosse deslocada pelo `[BOS]`, e por que esse número não diria nada sobre a qualidade do modelo?
5. Na curva de deleção, o que significa a curva ficar perto de 1 mesmo com 100% do contexto removido?
6. Rode o notebook `{outro}` com a mesma configuração e compare os bits por byte. Que conclusão a leitura assimétrica da caixa 9 permite tirar?"""
    else:
        exercicios = r"""
1. Na caixa 0, defina `CORPUS = "B"` e rode tudo de novo. O que muda no código e o que continua igual?
2. Na caixa 7, baixe `T_gen` para 8. O que acontece com o texto, e por quê?
3. Na caixa 0, dobre `EPOCAS_AULA`. Os bits por byte de validação caem na mesma proporção que a perda de treino?
4. Na caixa 6, por que a perda divide a soma de cada janela pelo seu próprio $l$, e não a soma do lote inteiro pelo total de máscaras do lote?
5. Na curva de deleção, o que significa a curva ficar perto de 1 mesmo com 100% do contexto removido?
6. Rode o notebook `{outro}` com a mesma configuração e compare os bits por byte. Que conclusão a leitura assimétrica da caixa 9 permite tirar?"""
    c.append(md("## Exercícios\n" + exercicios.replace("{outro}", outro) + r"""

## Referências

A lista completa de referências, todas de acesso aberto, está em `referencias/REFERENCIAS.md`, no repositório."""))
    return c


IGUAL = "Nada. Esta caixa é idêntica nos dois notebooks."
NOTAS = {
    0: ('A configuração é a mesma, exceto por `MODO`, que aqui vale `"autoregressivo"` e no notebook da difusão vale `"difusao"`. A técnica traz a vazão medida deste modo.',
        'A configuração é a mesma, exceto por `MODO`, que aqui vale `"difusao"` e no notebook autorregressivo vale `"autoregressivo"`. A técnica traz a vazão medida deste modo.'),
    1: (IGUAL, IGUAL),
    2: (IGUAL, IGUAL),
    3: ("Nada. O preenchimento previsto existe para tornar justa a comparação entre os dois modos, e por isso é igual nos dois notebooks.",
        "Nada. O preenchimento previsto existe para tornar justa a comparação entre os dois modos, e por isso é igual nos dois notebooks."),
    4: ("É nesta caixa que os modos se separam. Aqui a janela fica intacta, entra deslocada por um `[BOS]` e é lida com máscara causal. No notebook da difusão, a janela entra sem deslocamento, com $l$ posições trocadas por `[MASK]`, e a atenção vê a janela inteira.",
        "É nesta caixa que os modos se separam. Aqui $l$ posições da janela viram `[MASK]`, e a atenção vê a janela inteira. No notebook autorregressivo, a janela fica intacta, entra deslocada por um `[BOS]` e é lida com máscara causal."),
    5: ("A arquitetura é a mesma, com o mesmo número de parâmetros, mas cada notebook treina os seus próprios pesos do zero. Dentro da rede, a única diferença é a máscara de atenção, causal aqui e ausente no notebook da difusão ($M = 0$ no bloco de atenção). Fora dela, mudam a entrada (caixa 4) e a perda (caixa 6).",
        "A arquitetura é a mesma, com o mesmo número de parâmetros, mas cada notebook treina os seus próprios pesos do zero. Dentro da rede, a única diferença é a máscara de atenção, ausente aqui ($M = 0$ no bloco de atenção) e causal no notebook autorregressivo. Fora dela, mudam a entrada (caixa 4) e a perda (caixa 6)."),
    6: ("A perda é outra. Aqui ela é a log-verossimilhança negativa exata da janela. No notebook da difusão ela é um estimador não enviesado da NELBO, que é um limite superior dessa quantidade, com peso $L/l$ por janela. A otimização (AdamW, cronograma da taxa, fp16 com GradScaler) é a mesma.",
        "A perda é outra. Aqui ela é um estimador não enviesado da NELBO, um limite superior da log-verossimilhança negativa, com peso $L/l$ por janela. No notebook autorregressivo ela é a log-verossimilhança negativa exata. A otimização (AdamW, cronograma da taxa, fp16 com GradScaler) é a mesma."),
    7: ("A geração é outra. Aqui sai um token por vez, da esquerda para a direita, com uma passagem da rede por token. No notebook da difusão, a janela começa toda mascarada e é revelada em `T_gen` passos, em ordem aleatória. O sorteio em float64 é o mesmo.",
        "A geração é outra. Aqui a janela começa toda mascarada e é revelada em `T_gen` passos, em ordem aleatória, e `T_gen` pode ser menor que o número de tokens. No notebook autorregressivo sai um token por vez, da esquerda para a direita, com uma passagem da rede por token. O sorteio em float64 é o mesmo."),
    8: ("A fórmula dos bits por byte é a mesma, mas o significado muda. Aqui o número é exato. No notebook da difusão ele vem da NELBO e é um limite superior, o que torna a comparação assimétrica. A calibração por temperatura usa a mesma rotina, sobre populações diferentes, todas as posições no modo autorregressivo e só as mascaradas na difusão.",
        "A fórmula dos bits por byte é a mesma, mas o significado muda. Aqui o número vem da NELBO e é um limite superior. No notebook autorregressivo ele é exato, o que torna a comparação assimétrica. A calibração por temperatura usa a mesma rotina, sobre populações diferentes, todas as posições no modo autorregressivo e só as mascaradas na difusão."),
    9: ("Nada. O intervalo entre obras e a regra de comparação entre os modos são os mesmos nos dois notebooks.",
        "Nada. O intervalo entre obras e a regra de comparação entre os modos são os mesmos nos dois notebooks."),
    10: ("O código é idêntico. Só este notebook traz a medida de memorização com muitas épocas, que foi feita neste modo.",
         "O código é idêntico. A medida de memorização com muitas épocas aparece só no notebook autorregressivo, porque foi feita naquele modo."),
}


def do_modo(itens: list, modo: str) -> list:
    return [(tipo, texto) for tipo, texto, modos in itens if modos in ("ambos", modo)]


def tecer(cels: list[dict], modo: str) -> list[dict]:
    """Give every box the three parts: theory (the box text and its building blocks from
    notebooks/teoria.py), application (its code) and technique (problems and solutions from
    notebooks/tecnica.py, inserted before the next box starts). Blocks are numbered per box."""
    def cel(tipo, texto):
        return md(texto) if tipo == "md" else code(texto)

    def tecnica(caixa):
        itens = do_modo(TECNICA.get(caixa, []), modo)
        return [md("### Técnica")] + [cel(t, x) for t, x in itens] if itens else []

    saida, caixa, aplicacao_aberta = [], None, False
    for c in cels:
        m = re.match(r"## (Caixa (\d+)\.|Exercícios)", c["source"]) if c["cell_type"] == "markdown" else None
        if m:
            if caixa is not None:
                saida += tecnica(caixa)
            caixa = int(m.group(2)) if m.group(2) else None
            if caixa is not None:
                cabeca, _, resto = c["source"].partition("\n")
                outro = "da difusão" if modo == "autoregressivo" else "autorregressivo"
                nota = NOTAS[caixa][modo != "autoregressivo"]
                c = dict(c, source=f"{cabeca}\n\n> **O que muda em relação ao notebook {outro}.** {nota}\n\n### Teoria\n{resto}")
                aplicacao_aberta = True
        elif c["cell_type"] == "code" and caixa is not None and aplicacao_aberta:
            saida.append(md("### Aplicação"))
            aplicacao_aberta = False
        saida.append(c)
        if m and caixa is not None:
            k = iter(range(1, 100))
            for t, x in do_modo(TEORIA.get(caixa, []), modo):
                saida.append(cel(t, re.sub(r"\*\*Bloco #,", lambda _: f"**Bloco {caixa}.{next(k)},", x)))
    return saida


def main():
    for modo, nome in ARQUIVOS.items():
        nb = {"cells": tecer(celulas(modo), modo),
              "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                           "kernelspec": {"display_name": "Python 3", "name": "python3"},
                           "language_info": {"name": "python"}},
              "nbformat": 4, "nbformat_minor": 5}
        for i, c in enumerate(nb["cells"]):
            c["id"] = f"c{i:03d}"
        (PASTA / nome).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{nome}: {len(nb['cells'])} células")


if __name__ == "__main__":
    main()

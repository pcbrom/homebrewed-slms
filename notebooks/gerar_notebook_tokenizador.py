#!/usr/bin/env python3
"""Build the tokenizer notebook, notebooks/slm_tokenizador.ipynb.

Seven parts, each told as theory (building blocks), application (the code, executed) and technique
(problems the project met and the measurement that revealed them): the tokenizer contract, bytes,
BPE by hand, the choice of V and of the pre-tokenization pattern, a tokenizer of one's own plugged
into the contract (the Morse-style code of docs/decisao_tokenizador_codigo.md), training both
architectures with the tokenizers built here at equal compute, and the fair reading of the result
against the full experiment in resultados/m10/.

    python notebooks/gerar_notebook_tokenizador.py
"""
import json
import sys
from pathlib import Path

PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA))
from gerar_notebook import REPO, code, md  # noqa: E402

ARQUIVO = "slm_tokenizador.ipynb"


def celulas() -> list[dict]:
    c = []
    c.append(md(r"""
# Tokenizador do zero, e um tokenizador seu no lugar dele

Todo modelo de linguagem lê números, não letras. O tokenizador é a peça que transforma texto em números e números de volta em texto, e a escolha dele decide o comprimento das sequências, o tamanho da camada de saída e, com isso, quanto custa cada época de treino. Este notebook constrói tokenizadores do zero, do mais simples ao mais elaborado, mostra como encaixar um tokenizador inventado por você no mesmo pacote e treina com eles as duas arquiteturas do projeto, a autorregressiva e a de difusão mascarada.

Cada parte é contada em três tempos. A teoria vem decomposta em blocos de construção matemáticos, a aplicação roda o código e mostra o que entra e o que sai, e a técnica traz os problemas que o projeto encontrou e a medida que os revelou. Os notebooks irmãos, `slm_autorregressivo.ipynb` e `slm_difusao.ipynb`, seguem o modelo inteiro caixa a caixa. Aqui o foco é a caixa 2, o tokenizador, e o que muda no resto quando ela muda.

Projeto, decisões e referências em [{REPO}]({REPO}). Para rodar no Colab, escolha **Ambiente de execução, Alterar o tipo de ambiente de execução, GPU T4** e execute as células em ordem. As corridas da parte 6 levam poucos minutos numa GPU de mesa e mais numa T4.

Se este material for útil para você, deixe uma estrela no repositório e cite-o pelo `CITATION.cff`.
""".replace("{REPO}", REPO)))
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
sys.path.insert(0, os.path.join(RAIZ, "experimentos", "m10_codigo"))
os.chdir(RAIZ)
import torch
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
EPOCAS_REF = float(os.environ.get("SLM_EPOCAS", "16"))
print("dispositivo:", DISPOSITIVO, torch.cuda.get_device_name(0) if DISPOSITIVO == "cuda" else "", "| épocas do BPE de referência:", EPOCAS_REF)
""".replace("{REPO}", REPO)))
    c.append(md(r"""
## Mapa matemático

| Símbolo | Significado |
|---|---|
| $d$, $B_d$ | um documento (aqui, um capítulo) e o número de bytes dele em UTF-8, depois da normalização NFC |
| $\tau$, $\tau^{-1}$ | o tokenizador, de texto para ids, e o caminho de volta, de ids para bytes |
| $V$ | tamanho do vocabulário, com os ids 0 a 3 reservados para `[PAD]`, `[BOS]`, `[EOS]` e `[MASK]` |
| $n_d = \lvert\tau(d)\rvert$ | número de símbolos do documento |
| $\sigma = \sum_d n_d / \sum_d B_d$ | símbolos por byte, a densidade do tokenizador |
| $\mathrm{NLL}(d)$ | perda do modelo no documento, a log-verossimilhança negativa em nats |
| $\mathrm{bpb}$ | bits por byte, a régua que não depende do tokenizador |
| $L$, $n_c$, $h$ | janela de contexto em símbolos, número de camadas e largura da rede |
| $C(V, L)$ | custo de treino por símbolo, em FLOPs |
| $A$, $\ell_u$, $H_A$ | alfabeto de um código, comprimento da palavra de código da unidade $u$ e entropia em base $A$ (parte 5) |
| $S_b$, $E_b$ | símbolos de treino e épocas da corrida com o tokenizador $b$ (parte 6) |

A cadeia é curta. O texto $d$ vira $\tau(d)$, o modelo aprende sobre $\tau(d)$, e a medida volta ao texto pelo número de bytes, para que tokenizadores diferentes fiquem na mesma escala.

Os textos vêm do corpus A, a prosa de Machado de Assis na grafia das edições de época, já limpa e dividida em treino, validação e teste no repositório. Ao longo do notebook, texto retido é o que não entra no treino, aqui a validação.
"""))
    c.append(code(r"""
from slm.corpus import ler_documentos
treino_docs, valid_docs = ler_documentos("A", "treino"), ler_documentos("A", "validacao")
treino = [d["texto"] for d in treino_docs]
valid = [d["texto"] for d in valid_docs]
B_treino, B_valid = sum(d["bytes"] for d in treino_docs), sum(d["bytes"] for d in valid_docs)
print(f"treino: {len(treino)} capítulos, {B_treino / 1e6:.2f} MB | validação: {len(valid)} capítulos, {B_valid / 1e6:.2f} MB")
FRASE = "Capitú tinha olhos de ressaca, disse-lhe elle n'aquella hora."
"""))

    # Part 1, the contract
    c.append(md(r"""
## Parte 1. O contrato

### Teoria

Um tokenizador do projeto é qualquer objeto que cumpra um contrato curto. É ele que permite trocar o tokenizador sem mexer em nenhuma outra caixa do pacote.

**Bloco 1.1, ida e volta.** $\tau: \text{texto} \to \{4, \dots, V-1\}^*$ e $\tau^{-1}: \{0, \dots, V-1\}^* \to \text{bytes}$, com $\tau^{-1}(\tau(s)) = \mathrm{utf8}(s)$ para toda cadeia $s$. Nada se perde, nem acento, nem espaço, nem caractere raro.

**Bloco 1.2, os testes U1 a U6.** U1 é a ida e volta exata, U2 o determinismo (o mesmo texto dá sempre os mesmos ids, e a linha "U2 persistência" confere o mesmo depois de salvar e carregar o tokenizador), U3 os ids dentro de $\{0, \dots, V-1\}$, U4 os especiais nunca produzidos a partir de texto, mesmo quando o texto contém a cadeia `[MASK]`, U5 o decode de qualquer id isolado sem erro, e U6 a exigência de que nada disso pressuponha BPE. Código em `conferir_contrato`, no `slm/tokenizador.py`.

**Bloco 1.3, a régua.** Os bits por byte de um modelo são

$$\mathrm{bpb} = \frac{\sum_d \mathrm{NLL}(d)}{\ln 2 \cdot \sum_d B_d} = \underbrace{\frac{\sum_d \mathrm{NLL}(d)}{\ln 2 \cdot \sum_d n_d}}_{\text{bits por símbolo}} \cdot \underbrace{\frac{\sum_d n_d}{\sum_d B_d}}_{\sigma}.$$

Um tokenizador que encurta a sequência baixa $\sigma$ e, em troca, deixa cada símbolo mais difícil de prever. Só o produto compara tokenizadores, e por isso nenhuma tabela deste notebook compara perda por token entre tokenizadores diferentes.
"""))
    c.append(md("### Aplicação"))
    c.append(code(r"""
import tempfile
from slm.tokenizador import TokenizadorBytes, conferir_contrato
sonda = valid[:3] + [FRASE, "a ação do coração \u2014 漢字 \U0001F642 [MASK]"]
bytes_tok = TokenizadorBytes()
for teste, ok in conferir_contrato(bytes_tok, sonda, tempfile.mkdtemp()).items():
    print(f"{teste:32s} {'passa' if ok else 'FALHA'}")
"""))
    c.append(md(r"""
### Técnica

**Mesma letra, bytes diferentes.** O Unicode admite duas formas para a mesma letra acentuada. Na forma composta (NFC), "ã" é um único ponto de código. Na forma decomposta (NFD), é um "a" seguido de um til combinante. A expressão regular de pré-tokenização separa a marca combinante da letra, e a palavra se desfaz em pedaços. O corpus do projeto é normalizado em NFC uma única vez, antes de tudo, e os bytes da régua são contados sobre esse texto.
"""))
    c.append(code(r"""
import unicodedata, regex
from slm.tokenizador import GPT2_REGEX
for forma in ("NFC", "NFD"):
    t = unicodedata.normalize(forma, "a ação do coração")
    print(forma, len(t.encode()), "bytes |", regex.findall(GPT2_REGEX, t))
"""))

    # Part 2, bytes
    c.append(md(r"""
## Parte 2. Bytes, o tokenizador mais simples

### Teoria

**Bloco 2.1, alfabeto.** Os 256 valores de um byte mais os 4 especiais dão $V = 260$. Como todo texto é uma sequência de bytes, todo texto tem representação, e não existe token desconhecido.

**Bloco 2.2, densidade.** Cada byte vira um símbolo, então $\sigma = 1$. Uma janela de $L$ símbolos enxerga exatamente $L$ bytes de texto. É o ponto de partida contra o qual os outros tokenizadores medem o quanto encurtam a sequência.
"""))
    c.append(md("### Aplicação"))
    c.append(code(r"""
ids = bytes_tok.encode(FRASE)
print(len(FRASE), "caracteres,", len(FRASE.encode()), "bytes,", len(ids), "ids")
print(ids[:24], "...")
print("o 'ú' de Capitú ocupa dois ids:", bytes_tok.encode("ú"))
"""))
    c.append(md(r"""
### Técnica

**Contexto menor na mesma janela.** Com $L = 256$, o modelo sobre bytes vê 256 bytes de texto, e o modelo sobre o BPE do projeto vê cerca de 911. Uma comparação justa entre os dois precisa decidir se iguala a janela em símbolos ou o contexto em bytes, e o experimento do projeto mediu as duas formas lado a lado. A parte 7 mostra que a escolha pesa. A 16 épocas, os bytes perdem 0,80 bit por byte para o BPE com a mesma janela em símbolos e 1,41 com o mesmo contexto em bytes.
"""))

    # Part 3, BPE by hand
    c.append(md(r"""
## Parte 3. BPE construído à mão

### Teoria

O BPE em nível de byte parte do alfabeto de bytes e funde, repetidas vezes, o par vizinho mais frequente num símbolo novo (Sennrich et al., 2016).

**Bloco 3.1, pré-tokenização.** Uma expressão regular corta o texto em pedaços, como palavras com o espaço que as precede, números e pontuação, e nenhuma fusão atravessa a fronteira entre dois pedaços. O padrão do GPT-2 (Radford et al., 2019) é o usado no projeto.

**Bloco 3.2, contagem de pares.** Depois da pré-tokenização, o corpus vira um multiconjunto de pedaços $w$ com frequência $f(w)$, e para cada par vizinho $(a, b)$, $c(a, b) = \sum_w f(w)\, \#\{(a, b) \text{ vizinhos em } w\}$.

**Bloco 3.3, fusão e desempate.** Escolhe-se $(a^*, b^*) = \arg\max c(a, b)$, com o empate decidido pelo menor par de ids, cria-se o símbolo $a^* b^*$ com o próximo id livre e reescrevem-se todos os pedaços. Repete-se até haver $V$ símbolos.

**Bloco 3.4, codificação.** A codificação aplica as fusões na ordem em que foram aprendidas, e a decodificação concatena os bytes de cada símbolo. Como toda fusão concatena bytes, o contrato da parte 1 vale sem esforço adicional. Código em `BPEDidatico`, no `slm/tokenizador.py`.
"""))
    c.append(md("### Aplicação"))
    c.append(code(r"""
import collections
from slm.tokenizador import ORDEM_BYTES, BPEDidatico, BPEBiblioteca
print("pré-tokenização:", regex.findall(GPT2_REGEX, FRASE))
amostra = treino[:20]
id_byte = {bytes([b]): 4 + i for i, b in enumerate(ORDEM_BYTES)}
pedacos = collections.Counter(tuple(bytes([b]) for b in p.encode()) for t in amostra for p in regex.findall(GPT2_REGEX, t))
cont = collections.Counter()
for p, f in pedacos.items():
    for par in zip(p, p[1:]):
        cont[par] += f
a_mao = max(cont, key=lambda par: (cont[par], -id_byte[par[0]], -id_byte[par[1]]))
didatico = BPEDidatico().treinar(amostra, 260 + 60)
print("primeira fusão à mão:", a_mao, "com", cont[a_mao], "ocorrências | pelo pacote:", didatico.merges[0])
def mostrar(x):
    try:
        return repr(x.decode("utf-8"))
    except UnicodeDecodeError:
        return repr(x)
for a, b in didatico.merges[:12]:
    print(f"{mostrar(a):>10} + {mostrar(b):<10} -> {mostrar(a + b)}")
biblioteca = BPEBiblioteca().treinar(amostra, 260 + 60)
print("as 60 fusões da implementação didática e da biblioteca tokenizers são iguais:", didatico.merges == biblioteca.merges())
"""))
    c.append(md(r"""
### Técnica

**Duas implementações, uma resposta.** Na amostra de 440 KB do corpus A, em 3.463 das 3.836 fusões (90%) mais de um par empatou na contagem máxima, e o primeiro empate apareceu na fusão 56 (`scripts/medir_empates_bpe.py`). Nesses casos só a regra de desempate escolhe o par, e duas implementações sem regra escrita podem divergir. A versão didática adotou por escrito a convenção que reproduz a da biblioteca, e as duas ficaram idênticas, fusão a fusão e id a id, e um teste do pacote garante isso a cada mudança. A célula acima é a mesma verificação, numa amostra pequena.
"""))

    # Part 4, V and the pattern
    c.append(md(r"""
## Parte 4. Quantos símbolos, e como cortar o texto

### Teoria

**Bloco 4.1, tokens ausentes.** Para um BPE com vocabulário $V$, a fração de tokens aprendidos que nunca aparecem no texto retido é $a(V) = \lvert\{t \ge 260 : t \notin \tau(\text{retido})\}\rvert / (V - 260)$. Cada token ausente é uma linha do embedding treinada com pouquíssimos exemplos (Land e Bartolo, 2024).

**Bloco 4.2, a regra do projeto.** $V^* = \max\{V \in \{1.024, 2.048, 4.096, 8.192\} : a(V) \le 0{,}10\}$, decidida antes de qualquer treino de modelo. Código em `escolher_vocabulario`. O retido usado aqui é a validação, a mesma que depois mede os bits por byte, e a regra olha só para quais tokens aparecem nela, não para nenhum resultado de modelo.

**Bloco 4.3, o preço de $V$.** Na convenção de Kaplan et al. (2020), treinar custa cerca de 6 FLOPs por parâmetro por símbolo, 2 na ida e 4 na volta. Cada camada tem cerca de $12 h^2$ parâmetros (atenção e MLP), o que dá $72\, n_c\, h^2$, a atenção sobre a janela soma $6\, n_c\, L\, h$, e a camada de saída soma $6\, h\, V$:

$$C(V, L) = 72\, n_c\, h^2 + 6\, n_c\, L\, h + 6\, h\, V.$$

A contagem omite o embedding de entrada, que é uma consulta a uma tabela. Um $V$ maior baixa $\sigma$, mas encarece cada símbolo pelo último termo, e a parte 6 usa essa fórmula para igualar o orçamento.

**Bloco 4.4, o padrão de pré-tokenização.** O padrão do GPT-2 foi escrito para o inglês, com contrações como `'s` e `'re`. No português da época, a elisão aparece em "n'aquella" e "d'elle". Uma variante que mantém a elisão inteira no mesmo pedaço é medida abaixo, só como comparação, sem trocar o padrão do projeto.
"""))
    c.append(md("### Aplicação"))
    c.append(code(r"""
from slm.tokenizador import escolher_vocabulario, fracao_ausente
V_escolhido, ausentes = escolher_vocabulario(treino, valid)
print(" V     tokens ausentes  símbolos por byte")
for V, a in ausentes.items():
    tok_V = BPEBiblioteca().treinar(treino, V)
    s = sum(len(tok_V.encode(t)) for t in valid) / B_valid
    print(f"{V:5d}  {a:14.3f}  {s:17.3f}", "<- escolhido" if V == V_escolhido else "")
bpe = BPEBiblioteca().treinar(treino, V_escolhido)
bpe_repo = BPEBiblioteca.carregar("tokenizador/A")
print("\nigual ao tokenizador do repositório:", bpe.merges() == bpe_repo.merges())
"""))
    c.append(code(r"""
PT_REGEX = r" ?\p{L}+(?:['’]\p{L}+)*| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
print("GPT-2:", regex.findall(GPT2_REGEX, "disse-lhe elle n'aquella hora"))
print("PT   :", regex.findall(PT_REGEX, "disse-lhe elle n'aquella hora"))
bpe_pt = BPEBiblioteca(regex_pt=PT_REGEX).treinar(treino, V_escolhido)
for nome, t in (("GPT-2", bpe), ("PT", bpe_pt)):
    s = sum(len(t.encode(x)) for x in valid) / B_valid
    print(f"{nome:5s} | símbolos por byte {s:.4f} | tokens ausentes {fracao_ausente(t, valid):.3f}")
"""))
    c.append(md(r"""
### Técnica

**Vocabulário grande demais para o texto.** Com $V = 8.192$, 12% dos tokens aprendidos no corpus A nunca aparecem na validação, e a regra escolhe $V = 4.096$. No corpus B ela escolheu $V = 1.024$, com um texto retido de 174 KB contra 540 KB do corpus A, e um retido menor deixa mais tokens sem aparecer, o que empurra $V$ para baixo. Na análise de sensibilidade do projeto, treinar com $V = 2.048$ ou $8.192$ não mudou o sinal da comparação entre os dois modos, então a regra serve para evitar desperdício, não para decidir o resultado.

**Medir antes de adotar.** Um padrão de pré-tokenização novo só entraria no projeto com bits por byte de modelos treinados, nunca pela contagem de símbolos da célula acima.
"""))

    # Part 5, a tokenizer of one's own
    c.append(md(r"""
## Parte 5. Um tokenizador seu no contrato

### Teoria

Qualquer tokenizador que cumpra o contrato serve ao pacote, mesmo um que não tenha nada de BPE. O exemplo aqui é uma proposta do autor do projeto, descrita em `docs/decisao_tokenizador_codigo.md`. O modelo não aprende sobre o texto, e sim sobre uma língua de código $C$, desenhada para carregar mais informação por bit, no estilo do código Morse, em que a letra mais frequente, o "E", recebe o sinal mais curto.

**Bloco 5.1, unidades.** As 4.096 pré-palavras mais frequentes do treino (os pedaços da pré-tokenização do bloco 3.1) ficam inteiras, e as demais são cortadas por uma regra de morfemas escrita à mão para o português (`experimentos/m10_codigo/morfema.py`), em radical, sufixo derivacional e flexão.

**Bloco 5.2, código de Huffman de base $A$.** Cada unidade $u$, com frequência $p_u$ no treino, recebe uma palavra de código sobre um alfabeto de $A$ símbolos, livre de prefixo, de comprimento $\ell_u$. O código de Huffman minimiza $\bar\ell = \sum_u p_u \ell_u$ entre os códigos livres de prefixo, e $H_A \le \bar\ell < H_A + 1$, com $H_A = -\sum_u p_u \log_A p_u$. A tabela é derivada do treino e congelada antes de qualquer treino de modelo.

**Bloco 5.3, escape.** Uma unidade ausente da tabela é soletrada entre dois escapes, caractere a caractere, e por bytes quando o caractere nunca foi visto. Assim $\tau^{-1}(\tau(s)) = \mathrm{utf8}(s)$ para qualquer texto, e o contrato vale.

**Bloco 5.4, as três frentes.** A proposta busca ao mesmo tempo menos símbolos por byte, mais informação por símbolo e menos bits por byte, e as três se ligam pela identidade bits nominais por byte $= \sigma \log_2 A$. Um alfabeto pequeno dá símbolos baratos e sequências longas, e um grande, o contrário. É o tamanho do alfabeto que decide qual frente se ganha e qual se paga.
"""))
    c.append(md("### Aplicação"))
    c.append(code(r"""
import hashlib, json, time
from codigo_morse import CodigoMorse
t0 = time.time()
codigo = CodigoMorse.construir(treino, A=256, n_topo=4096, variante="prefixo")
pasta_codigo = tempfile.mkdtemp()
contrato = conferir_contrato(codigo, sonda, pasta_codigo)
print(f"código construído em {time.time() - t0:.0f} s | inventário de {len(codigo.inventario)} unidades | V = {codigo.vocab_size}")
for teste, ok in contrato.items():
    print(f"{teste:32s} {'passa' if ok else 'FALHA'}")
congelado = json.load(open("tokenizador/m10_C1/metadados.json"))["sha256"]
print("tabela igual à congelada no experimento:", hashlib.sha256(open(f"{pasta_codigo}/tabela.json", "rb").read()).hexdigest() == congelado)
"""))
    c.append(code(r"""
import math
from codigo_morse import unidades
for frase in (FRASE, "As desconfianças do sobrinho, e um Zwqx que ninguém viu."):
    us = unidades(frase, codigo.topo, "prefixo")
    print("unidades:", us)
    print("palavras de código:", [codigo.cod.get(u, "escape e soletração") for u in us])
print("símbolos do código por byte: treino", round(sum(len(codigo.encode(x)) for x in treino) / B_treino, 3),
      "| validação", round(sum(len(codigo.encode(x)) for x in valid) / B_valid, 3))
print(" tokenizador     V  símbolos por byte  bits nominais por byte")
for nome, t in (("bytes", bytes_tok), ("BPE", bpe), ("código", codigo)):
    s = sum(len(t.encode(x)) for x in valid) / B_valid
    print(f"{nome:8s} {t.vocab_size:6d}  {s:17.3f}  {s * math.log2(t.vocab_size - 4):22.2f}")
"""))
    c.append(md(r"""
### Técnica

**O separador custa caro.** No Morse, as fronteiras entre letras são pausas. Nas medidas do projeto, sobre o inventário de unidades mistas com a aproximação de sufixos, marcar cada fronteira com um símbolo próprio deixou o código 62,8% mais caro em bits por byte que a versão livre de prefixo (5,30 contra 3,26), e por isso o código desta parte não usa separador.

**O inventário se ajusta ao treino.** O código é mais denso no treino (0,339 símbolos por byte) que na validação (0,401), porque palavras fora da tabela pagam escape e soletração. O fator de épocas da parte 6 usa a densidade do treino, que é a que consome computação.

**Densidade não é aprendizado.** Nas medidas sem contexto, as unidades do BPE têm entropia por byte menor que os inventários de morfemas testados, 2,75 bits contra 2,84 do melhor deles e 2,92 do inventário desta parte. Contar símbolos não diz se um modelo aprende melhor sobre o código, e só o treino da parte 6 responde.
"""))

    # Part 6, training both architectures
    c.append(md(r"""
## Parte 6. Treino das duas arquiteturas com os tokenizadores construídos

### Teoria

**Bloco 6.1, orçamento igual.** Uma corrida com o tokenizador $b$ custa $S_b \cdot E_b \cdot C(V_b, L)$ FLOPs, com $S_b$ símbolos de treino e $E_b$ épocas. Para gastar o mesmo que o BPE de referência em $E_{ref}$ épocas,

$$E_b = E_{ref} \cdot \frac{S_{ref}\, C(V_{ref}, L)}{S_b\, C(V_b, L)}.$$

Um tokenizador menos denso tem mais símbolos e recebe menos passagens pelo texto.

**Bloco 6.2, o que se compara.** As corridas usam a mesma rede, os mesmos dados e os hiperparâmetros congelados no protocolo do projeto, e a medida é a do bloco 1.3, bits por byte do texto original na validação. No modo autorregressivo o número é exato. Na difusão é um limite superior, estimado aqui com um único sorteio de máscaras por janela ($K = 1$), correto em média, e uma desvantagem dela não diz nada sobre a verossimilhança verdadeira.
"""))
    c.append(md("### Aplicação"))
    c.append(code(r"""
def custo(V, L=256, n_c=6, h=384):
    return 72 * n_c * h * h + 6 * n_c * L * h + 6 * h * V
toks = {"BPE": bpe, "bytes": bytes_tok, "código": codigo}
S = {nome: sum(len(t.encode(x)) for x in treino) for nome, t in toks.items()}
fator = {nome: S["BPE"] * custo(bpe.vocab_size) / (S[nome] * custo(t.vocab_size)) for nome, t in toks.items()}
plano = json.load(open("resultados/m10/m10_plano.json"))["bracos"]
for nome, braco in (("BPE", "B1"), ("bytes", "B0"), ("código", "C1")):
    print(f"{nome:7s} | {S[nome] / B_treino:.3f} símbolos por byte de treino | fator de épocas {fator[nome]:.3f}"
          f" | no experimento {plano[braco]['fator_epocas']['simbolos']:.3f}")
"""))
    c.append(code(r"""
from slm.config import Config
from slm.corpus import tokenizar
from slm.janelas import janelar
from slm.modelo import ConfigRede, Transformer
from slm.treino import treinar
from slm.avaliacao import avaliar
HP = json.load(open("preregistro/m6_delta_min.json"))["escolha"]
L = 256

def corrida(nome, modo):
    tok = toks[nome]
    tr, va = tokenizar(treino_docs, tok), tokenizar(valid_docs, tok)
    janelas, _ = janelar([d["ids"] for d in tr], L)
    cfg = Config(modo=modo, vocab_size=tok.vocab_size, L=L, epocas=EPOCAS_REF * fator[nome], papel="aula", **HP[modo],
                 precisao="fp16" if DISPOSITIVO == "cuda" else "fp32",
                 atencao="efficient" if DISPOSITIVO == "cuda" else "math").validar()
    torch.manual_seed(cfg.semente)
    modelo = Transformer(ConfigRede(vocab_size=cfg.vocab_size, L=L, n_layer=cfg.n_layer, d=cfg.d, n_head=cfg.n_head,
                                    dropout=cfg.dropout)).to(DISPOSITIVO)
    t0 = time.time()
    treinar(cfg, modelo, janelas, DISPOSITIVO)
    modelo.eval()
    bpb = avaliar(modelo, va, modo, L, K=1, dispositivo=DISPOSITIVO)["bpb"]
    print(f"{nome:7s} {modo:15s} | {cfg.epocas:6.2f} épocas | {time.time() - t0:5.0f} s | bpb de validação {bpb:.4f}")
    return modelo, bpb

resultados, modelos = {}, {}
for nome, modo in (("BPE", "autoregressivo"), ("BPE", "difusao"), ("bytes", "autoregressivo"), ("código", "autoregressivo")):
    modelos[(nome, modo)], resultados[(nome, modo)] = corrida(nome, modo)
"""))
    c.append(code(r"""
from slm.geracao import gerar_ar
from slm.tokenizador import ESPECIAIS
EOS = ESPECIAIS["[EOS]"]
amostras = gerar_ar(modelos[("código", "autoregressivo")], L, n=3, semente=1, dispositivo=DISPOSITIVO)
for linha in amostras.tolist():
    ids = linha[: linha.index(EOS)] if EOS in linha else linha
    codigo.invalidos = 0
    texto = codigo.decode(ids).decode("utf-8", "replace")
    print(f"[{codigo.invalidos} palavras de código inválidas] {texto[:300]}\n")
"""))
    c.append(md(r"""
### Técnica

**Palavra de código pela metade.** A geração para no limite da janela, que pode cair no meio de uma palavra de código. A volta descarta esse resto, e no experimento, com o código desta parte, 6 e 8 de 32 amostras terminaram assim, a 16 e a 256 épocas. Uma palavra que não existe na tabela vira o caractere de substituição, e a contagem aparece no início de cada amostra acima.

**Uma semente não decide.** Os números desta parte vêm de uma semente cada. O experimento completo usou 3 sementes por braço, duas formas de janela e dois orçamentos, e é ele que a parte 7 lê.
"""))

    # Part 7, fair reading
    c.append(md(r"""
## Parte 7. Leitura justa

### Teoria

**Bloco 7.1, diferença pareada.** Com sementes pareadas entre braços, a diferença $\Delta_b = \mathrm{bpb}_b - \mathrm{bpb}_{ref}$ é estimada pela média sobre sementes, com intervalo de 95% pela distribuição $t$. Um intervalo inteiro acima de zero é desvantagem do braço $b$.

**Bloco 7.2, critério fixado antes.** A leitura pré-registrada, escrita e congelada antes das corridas, comparava o código com o BPE. Com 39 de 54 corridas vistas, a pergunta passou a ser a viabilidade do código, e essa emenda está registrada como feita depois de ver dados, com a leitura original relatada ao lado (`docs/decisao_tokenizador_codigo.md`).
"""))
    c.append(md("### Aplicação"))
    c.append(code(r"""
res_m10 = json.load(open("resultados/m10/m10_resultado.json"))
fmt = lambda v: f"{v['media']:+.3f} [{v['ic95'][0]:+.3f}, {v['ic95'][1]:+.3f}]"
print("diferença contra o BPE em bits por byte, média de 3 sementes e intervalo de 95%")
print("braço   janela igual em   16 épocas                256 épocas")
for braco, nome in (("C1", "código"), ("B0", "bytes")):
    for janela, rotulo in (("simbolos", "símbolos"), ("bytes", "bytes")):
        e16, e256 = (res_m10[f"autoregressivo|e{p}|{braco}-{janela}"] for p in (16, 256))
        print(f"{nome:7s} {rotulo:15s}   {fmt(e16):24s} {fmt(e256)}")
for nome in ("código", "bytes"):
    aqui = resultados[(nome, "autoregressivo")] - resultados[("BPE", "autoregressivo")]
    print(f"neste notebook, {nome}, janela igual em símbolos, {EPOCAS_REF:g} épocas do BPE, 1 semente: {aqui:+.3f}")
"""))
    c.append(md(r"""
### Técnica

**A ordem pode inverter com o orçamento.** Com computação igual, o modelo sobre bytes foi o pior braço a 16 épocas do BPE e o melhor a 256, cerca de 1 bit por byte abaixo do BPE. A comparação é contra um BPE que piorou de 1,954 para 2,849 bits por byte entre os dois orçamentos, porque decorou o texto de treino. Igualar a computação faz o modelo de bytes passar só 0,32 vez as épocas do BPE pelo texto, e ele decora menos. Essa explicação não foi medida à parte, e o controle entra no artigo do projeto.

**O resultado do código.** Pela leitura original, o código desta parte perde para o BPE nos dois orçamentos, por 0,745 bit por byte no melhor ponto, e as três variantes testadas no experimento perdem de 0,64 a 0,76. Pelos critérios de viabilidade, ele funciona. A ida e volta é exata, as gerações voltam a texto e, no ponto de 16 épocas, o modelo fica abaixo da entropia sem contexto das próprias unidades (2,70 contra 2,92 bits por byte). A 256 épocas fica acima, em 3,84, porque sobreajusta, como o BPE. Um resultado negativo medido com cuidado vale tanto quanto um positivo, e é ele que diz por onde não seguir.
"""))
    c.append(md(r"""
## Exercícios

1. Na parte 4, troque a grade de $V$ por `(512, 1024, 2048, 4096, 8192, 16384)`. Onde a fração de tokens ausentes passa de 10%, e o que isso diz sobre o tamanho do corpus?
2. Na parte 5, construa o código com `A=4096` em vez de 256. O que acontece com os símbolos por byte e com os bits nominais por byte?
3. Escreva um tokenizador seu, por exemplo um que codifique cada caractere pelo seu número de ponto de código, passe-o por `conferir_contrato` e treine com ele na parte 6.
4. Na parte 6, iguale o contexto em bytes em vez da janela em símbolos, com `L = round(256 * σ_b / σ_BPE)`, e recalcule o fator de épocas com `custo(V, L)` nessa janela, porque o termo da atenção muda. A ordem entre os tokenizadores muda?
5. Por que a parte 6 compara bits por byte e não a perda média por símbolo de cada corrida?

## Referências

A lista completa, com versões fixadas e licenças, está em `referencias/REFERENCIAS.md`, no repositório.
"""))
    return c


def main():
    nb = {"cells": celulas(),
          "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"c{i:03d}"
    (PASTA / ARQUIVO).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{ARQUIVO}: {len(nb['cells'])} células")


if __name__ == "__main__":
    main()

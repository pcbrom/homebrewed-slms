"""Mathematical building blocks of the notebooks.

mapa(modo) is the global map placed after the pipeline figure. TEORIA maps the box number to a
list of (kind, text, modes) cells placed right after the box introduction, with kind "md" or
"code" and modes "ambos", "autoregressivo" or "difusao". Block titles are written "**Bloco #,"
and numbered per notebook by the generator. Each block gives the definition, the derivation when
there is one, the package function that implements it and the link to the next box; code cells
redo most blocks by hand and check them against the package.
"""

AMBOS, AR, DIF = "ambos", "autoregressivo", "difusao"


def mapa(modo: str) -> str:
    if modo == DIF:
        transforma, entrada, condicional = r"\text{mascarar}", "z", r"p_\theta(\cdot \mid z)"
        linha_caixa4 = "| $l$, $S$, $z$ | número de máscaras, conjunto mascarado, janela mascarada | caixa 4, `slm/mascaramento.py` |"
        linha_perda = r"| $\hat\ell_{dif}(w)$ | estimativa da NELBO de uma janela, em nats | caixa 6, `slm/perda.py` |"
        perda, geracao = r"\hat\ell_{dif}(w)", "das máscaras ao texto, revelando posições em ordem aleatória"
    else:
        transforma, entrada, condicional = r"\text{deslocar}", "x", r"p_\theta(w_i \mid w_{<i})"
        linha_caixa4 = "| $x$ | janela deslocada por um BOS, lida com máscara causal | caixa 4, `slm/perda.py` |"
        linha_perda = r"| $\ell_{AR}(w)$ | log-verossimilhança negativa de uma janela, em nats | caixa 6, `slm/perda.py` |"
        perda, geracao = r"\ell_{AR}(w)", "um token por vez, da esquerda para a direita"
    return rf"""
## Mapa matemático, do texto à medida

Cada caixa recebe um objeto matemático e entrega outro. A tabela fixa a notação usada no notebook inteiro, e a cadeia logo abaixo mostra como os blocos se encaixam.

| Símbolo | Significado | Onde nasce |
|---|---|---|
| $d$, $B_d$ | um documento (capítulo ou lei) e o número de bytes dele em UTF-8, depois da NFC | caixa 1, `slm/dados.py` |
| $\tau(d) = (t_1, \dots, t_n)$ | ids dos tokens do documento, com $t_i \in \{{4, \dots, V-1\}}$ | caixa 2, `slm/tokenizador.py` |
| $w \in \{{0, \dots, V-1\}}^L$ | uma janela de $L$ tokens | caixa 3, `slm/janelas.py` |
{linha_caixa4}
| $f_\theta$, $p_\theta$ | rede com parâmetros $\theta$ e as distribuições condicionais que ela define | caixa 5, `slm/modelo.py` |
{linha_perda}
| $\mathrm{{NLL}}(d)$, $\mathrm{{bpb}}$ | perda de um documento e bits por byte | caixa 8, `slm/avaliacao.py` |
| IC | intervalo de confiança dos bits por byte | caixa 9, `slm/estatistica.py` |

Alguns símbolos carregam dois sentidos, e o contexto os separa. $d$ é documento e largura da rede, $S$ é o conjunto mascarado e a escala do GradScaler, $T$ é o número de passos da geração e a temperatura, $K$ é a matriz de chaves e o número de estimativas por janela, e $t$ é o tempo da difusão enquanto $t_i$ é um token. Os logits da rede são escritos $o$, para não se confundirem com a janela mascarada $z$.

$$
\underbrace{{d}}_{{\text{{caixa 1}}}}
\xrightarrow{{\ \tau\ }}
\underbrace{{(t_1,\dots,t_n)}}_{{\text{{caixa 2}}}}
\xrightarrow{{\ \oplus\,\mathrm{{EOS}},\ \text{{corte em }} L\ }}
\underbrace{{w}}_{{\text{{caixa 3}}}}
\xrightarrow{{\ {transforma}\ }}
\underbrace{{{entrada}}}_{{\text{{caixa 4}}}}
\xrightarrow{{\ f_\theta\ }}
\underbrace{{{condicional}}}_{{\text{{caixa 5}}}}
\xrightarrow{{\ -\log\ }}
\underbrace{{{perda}}}_{{\text{{caixa 6}}}}
\xrightarrow{{\ \sum_{{w \in d}}\ }}
\underbrace{{\mathrm{{NLL}}(d)}}_{{\text{{caixa 8}}}}
\xrightarrow{{\ \sum_d / (\ln 2 \sum_d B_d)\ }}
\underbrace{{\mathrm{{bpb}} \pm \mathrm{{IC}}}}_{{\text{{caixas 8 e 9}}}}
$$

O treino (caixa 6) fecha um laço sobre essa cadeia, porque o gradiente da perda em relação a $\theta$ volta à caixa 5 e altera $f_\theta$. A geração (caixa 7) percorre o mesmo $p_\theta$ no sentido contrário, {geracao}.
"""


TEORIA = {
    0: [("md", r"""
### Blocos de construção

**Bloco #, parâmetros sem o embedding.** Cada bloco do Transformer tem as projeções de atenção $Q$, $K$ e $V$ e a projeção de saída, que somam $4d^2$ pesos, além de um MLP de largura $4d$, com $8d^2$ pesos. Daí $N = 12\, n_{layer}\, d^2$, desprezando vieses e normalizações. Código em `scripts/dimensionar.py`.

**Bloco #, custo de um token de treino.** A passagem direta custa $2N$ nos produtos de matrizes das camadas, $2\, n_{layer}\, L\, d$ no produto $QK^\top$ da atenção e $2\, d\, V$ na projeção de saída. Essa é a convenção de Kaplan et al. (2020). O PaLM (Chowdhery et al., 2022) conta também o produto da atenção pelos valores, o que dobra o termo da atenção. A retropropagação custa cerca do dobro da passagem direta, e assim, na convenção de Kaplan,

$$C_{treino} = 6N + 6\, n_{layer}\, L\, d + 6\, d\, V.$$

**Bloco #, vazão e tempo.** Com pico $P$ e utilização MFU, a vazão é $P \cdot \mathrm{MFU} / C_{treino}$ tokens por segundo, e o tempo para ver $D$ tokens é $D$ dividido por ela. É esse número que decide quantas épocas e quantas sementes cabem no orçamento das caixas 6 e 9.
""", AMBOS), ("code", r"""
n_layer, d, L_, V_ = 6, 384, 256, 4096
N = 12 * n_layer * d * d
C = 6 * N + 6 * n_layer * L_ * d + 6 * d * V_
print(f"N = {N:,} | C_treino = {C / 1e6:.1f} MFLOP por token | {C / (6 * N) - 1:.1%} acima de 6N")
""", AMBOS)],
    1: [("md", r"""
### Blocos de construção

**Bloco #, normalização.** Todo texto passa por $u \mapsto \mathrm{NFC}(u)$, a forma canônica composta do Unicode. Depois dela, cada caractere acentuado do português é um único ponto de código.

**Bloco #, bytes.** $B_d = |\mathrm{UTF8}(\mathrm{NFC}(d))|$. É o denominador dos bits por byte da caixa 8, e por isso é contado aqui, uma única vez, e nunca sobre o texto que o tokenizador reconstrói.

**Bloco #, partição.** Cada obra é uma sequência de capítulos $c_1, \dots, c_m$. A divisão manda um primeiro bloco para o treino, um capítulo para a margem, um bloco contíguo para a validação, outro capítulo para a margem e o bloco final para o teste, cada bloco retido com pelo menos 5 capítulos e 30 KB. *Papéis avulsos*, com 12 contos, comporta só o bloco de validação e fica fora do teste. A partição é função do documento e fica fixada antes de qualquer treino. Código em `slm/dados.py`.
""", AMBOS), ("code", r"""
import unicodedata
from slm.corpus import ler_documentos
docs = ler_documentos(CORPUS)
d0 = docs[0]
print(d0["doc_id"], "| já em NFC:", unicodedata.is_normalized("NFC", d0["texto"]),
      "| B_d recontado:", len(d0["texto"].encode("utf-8")), "= bytes guardados:", d0["bytes"])
""", AMBOS)],
    2: [("md", r"""
### Blocos de construção

**Bloco #, alfabeto.** O vocabulário começa com os 256 bytes. Como todo texto é uma sequência de bytes, todo texto pode ser representado, e não existe token desconhecido.

**Bloco #, contagem de pares.** Depois da pré-tokenização, o corpus vira um multiconjunto de palavras $w$ com frequência $f(w)$. Para cada par adjacente $(a, b)$, $c(a,b) = \sum_w f(w)\, \#\{(a,b) \text{ adjacentes em } w\}$.

**Bloco #, fusão.** Escolhe-se $(a^*, b^*) = \arg\max c(a,b)$, com o empate decidido pelo menor par de ids, cria-se o símbolo $a^* b^*$ com o próximo id e reescrevem-se todas as palavras. Repete-se até haver $V$ símbolos.

**Bloco #, codificação e decodificação.** A codificação aplica as fusões na ordem em que foram aprendidas, e a decodificação concatena os bytes de cada símbolo. Como toda fusão concatena bytes, $\mathrm{decode}(\mathrm{encode}(s)) = s$ para qualquer $s$. Código em `BPEDidatico.treinar`, `encode` e `decode`, no `slm/tokenizador.py`. A saída desta caixa, $\tau(d)$, é a entrada da caixa 3.
""", AMBOS), ("code", r"""
import collections, regex
from slm.tokenizador import GPT2_REGEX, ORDEM_BYTES, BPEDidatico
id_byte = {bytes([b]): i for i, b in enumerate(ORDEM_BYTES)}
amostra2 = [d["texto"] for d in docs if d["particao"] == "treino"][:5]
palavras = collections.Counter(tuple(bytes([b]) for b in p.encode()) for t in amostra2 for p in regex.findall(GPT2_REGEX, t))
cont = collections.Counter()
for p, f in palavras.items():
    for par in zip(p, p[1:]):
        cont[par] += f
a_mao = max(cont, key=lambda par: (cont[par], -id_byte[par[0]], -id_byte[par[1]]))
pelo_pacote = BPEDidatico().treinar(amostra2, 265).merges[0]
print("primeira fusão à mão:", a_mao, "| pelo pacote:", pelo_pacote, "| iguais:", a_mao == pelo_pacote, "| contagem:", cont[a_mao])
""", AMBOS)],
    3: [("md", r"""
### Blocos de construção

**Bloco #, fim do documento.** $s(d) = \tau(d) \oplus (\text{EOS})$, de comprimento $n + 1$.

**Bloco #, janelas.** São $N_w(d) = \lceil (n+1)/L \rceil$ janelas, a janela $k$, contada a partir de zero, com os tokens $s_{kL+1}, \dots, s_{(k+1)L}$ e a última completada com EOS. Nenhum contexto atravessa de uma janela para outra.

**Bloco #, preenchimento.** $\phi = \dfrac{\sum_d \big(L\, N_w(d) - n_d - 1\big)}{\sum_d L\, N_w(d)}$ é a fração de posições que custam computação sem carregar texto. Código em `slm/janelas.py`. Cada janela $w$ segue para a caixa 4.
""", AMBOS), ("code", r"""
import math
from slm.janelas import janelas_documento
ids0 = tok.encode(docs[0]["texto"])
n = len(ids0)
Nw = math.ceil((n + 1) / 256)
a_mao = ids0 + [2] * (Nw * 256 - n)
print("n =", n, "| N_w = ceil((n+1)/L) =", Nw, "| igual ao pacote:", janelas_documento(ids0, 256).flatten().tolist() == a_mao)
""", AMBOS)],
    4: [("md", r"""
### Blocos de construção

**Bloco #, processo direto absorvente.** Com $\alpha_t = 1 - t$, cada token sobrevive com probabilidade $1 - t$ ou vira `[MASK]` com probabilidade $t$, de forma independente, ou seja, $q(z_i \mid w_i) = (1-t)\,\delta_{w_i}(z_i) + t\,\delta_{\mathrm{MASK}}(z_i)$.

**Bloco #, do tempo ao número de máscaras.** Dado $t$, o número de máscaras é $l \sim \mathrm{Bin}(L, t)$, e dado $l$ o conjunto $S$ é uniforme entre os $\binom{L}{l}$ subconjuntos. A caixa 6 mostra por que basta sortear $l$ uniforme em $\{1, \dots, L\}$.

**Bloco #, sorteio estratificado.** Com $U \sim U(0,1)$ comum ao lote e uma permutação $\pi$, $u_b = (\pi(b) - 1 + U)/B$ e $l_b = 1 + \lfloor u_b L \rfloor$. Cada $u_b$ cai num estrato diferente de $[0,1)$, o que não introduz viés e em geral reduz a variância da média do lote em relação ao sorteio independente, embora essa redução não seja garantida para toda função de $l$.

**Bloco #, um conjunto uniforme de tamanho $l$.** Sorteiam-se $L$ números uniformes e tomam-se como $S$ as $l$ posições de menor posto. Por simetria, todo subconjunto de tamanho $l$ tem a mesma probabilidade. Então $z = w$ fora de $S$ e $z = \mathrm{MASK}$ em $S$. Código em `sortear_l`, `mascara_de_l` e `mascarar`, no `slm/mascaramento.py`. A janela mascarada $z$ segue para a rede da caixa 5.
""", DIF), ("code", r"""
from slm.mascaramento import sortear_l, mascara_de_l
g4 = torch.Generator().manual_seed(1)
l4 = sortear_l(8, 256, g4)
m4 = mascara_de_l(l4, 256, g4)
print("l sorteados:", l4.tolist())
print("máscaras por linha iguais a l:", torch.equal(m4.sum(1), l4), "| um l por estrato de 1/8:", sorted(((l4 - 1) * 8 // 256).tolist()))
""", DIF), ("md", r"""
### Blocos de construção

**Bloco #, regra da cadeia.** Toda distribuição sobre janelas se escreve como $p(w) = \prod_{i=1}^{L} p(w_i \mid w_{<i})$, sem aproximação nenhuma. O modo autorregressivo modela cada fator.

**Bloco #, entrada deslocada.** A entrada é $x = (\text{BOS}, w_1, \dots, w_{L-1})$, e a saída na posição $i$ prevê $w_i$. O `[BOS]` ocupa o lugar do token que ainda não existe no início. Código em `entrada_ar`, no `slm/perda.py`.

**Bloco #, máscara causal.** Na atenção, $M_{ij} = 0$ se $j \le i$ e $-\infty$ caso contrário. Somada aos escores antes do softmax, ela zera o peso de toda posição futura, e a previsão de $w_i$ só vê $\text{BOS}, w_1, \dots, w_{i-1}$. A entrada $x$ segue para a rede da caixa 5.
""", AR), ("code", r"""
from slm.perda import entrada_ar
w4 = janelas_treino[:1, :8]
x4 = entrada_ar(w4)
print("janela  :", w4.tolist()[0])
print("entrada :", x4.tolist()[0])
print("deslocada por um BOS:", x4[0, 0].item() == 1 and torch.equal(x4[0, 1:], w4[0, :-1]))
""", AR)],
    5: [("md", r"""
### Blocos de construção

**Bloco #, entrada.** $h^{(0)}_i = E[x_i] + P_i$, com $E \in \mathbb{R}^{V \times d}$ a matriz de embedding e $P \in \mathbb{R}^{L \times d}$ as posições aprendidas.

**Bloco #, atenção.** Com $u = \mathrm{LN}(h)$, $Q = u W_Q$, $K = u W_K$ e $V_a = u W_V$, divididas em $H$ cabeças de tamanho $d_h = d/H$,

$$A = \mathrm{softmax}\!\Big(\frac{Q K^\top}{\sqrt{d_h}} + M\Big) V_a, \qquad h \leftarrow h + A W_O.$$

$M = 0$ na difusão e causal no modo autorregressivo. Essa é a única diferença entre os dois modos dentro da rede.

**Bloco #, MLP.** $h \leftarrow h + W_2\, \mathrm{GELU}(W_1\, \mathrm{LN}(h))$, de largura $4d$.

**Bloco #, saída.** $o = \mathrm{LN}(h^{(n_{layer})})\, E^\top \in \mathbb{R}^{L \times V}$, com a mesma $E$ da entrada. O logit de `[MASK]` é fixado em $-\infty$, e $p_\theta(\cdot \mid \text{entrada})_i = \mathrm{softmax}(o_i)$. Código em `Bloco.forward` e `Transformer.forward`, no `slm/modelo.py`. A caixa 6 transforma $p_\theta$ em perda.

A célula abaixo refaz a passagem direta de uma rede pequena com matrizes explícitas e confere o resultado contra o módulo.
""", AMBOS), ("code", r"""
import torch.nn.functional as F
from slm.modelo import ConfigRede, Transformer
torch.manual_seed(0)
mini = Transformer(ConfigRede(vocab_size=tok.vocab_size, L=16, n_layer=1, d=32, n_head=4)).eval()
x5 = janelas_treino[:2, :16]
b = mini.blocos[0]
causal5 = MODO == "autoregressivo"
with torch.no_grad():
    h = mini.emb(x5) + mini.pos[:16]
    q, k, v = b.qkv(b.ln1(h)).view(2, 16, 3, 4, 8).permute(2, 0, 3, 1, 4)
    escores = q @ k.transpose(-1, -2) / 8 ** 0.5
    if causal5:
        escores = escores.masked_fill(~torch.ones(16, 16, dtype=torch.bool).tril(), float("-inf"))
    h = h + b.proj((torch.softmax(escores, dim=-1) @ v).transpose(1, 2).reshape(2, 16, 32))
    h = h + b.out(F.gelu(b.fc(b.ln2(h))))
    z5 = mini.ln(h) @ mini.emb.weight.T
    z5[..., 3] = float("-inf")
    print("passagem direta à mão igual ao módulo:", torch.allclose(z5, mini(x5, causal5), atol=1e-5))
""", AMBOS)],
    6: [("md", r"""
### Blocos de construção

**Bloco #, entropia cruzada.** Para logits $o$ e alvo $y$, $-\log p(y) = -o_y + \log \sum_v e^{o_v}$. O gradiente em relação a $o$ é $p - e_y$, e a rede é empurrada a subir o logit do alvo e a baixar os demais na proporção da probabilidade que cada um recebe.

**Bloco #, perda autorregressiva.** $\ell_{AR}(w) = \sum_{i=1}^{L} -\log p_\theta(w_i \mid \text{BOS}, w_{<i})$. Pela regra da cadeia da caixa 4, isso é exatamente $-\log p_\theta(w)$, sem limite nem aproximação. No treino, a perda do lote é a média por token. Código em `perda_ar`, no `slm/perda.py`.
""", AR), ("md", r"""
### Blocos de construção

**Bloco #, entropia cruzada.** Para logits $o$ e alvo $y$, $-\log p(y) = -o_y + \log \sum_v e^{o_v}$. O gradiente em relação a $o$ é $p - e_y$, e a rede é empurrada a subir o logit do alvo e a baixar os demais na proporção da probabilidade que cada um recebe.

**Bloco #, NELBO da difusão em tempo contínuo.** Com o cronograma linear,

$$\mathrm{NELBO}(w) = \int_0^1 \frac{1}{t}\, \mathbb{E}_{z \sim q_t(\cdot \mid w)}\Big[\sum_{i:\, z_i = \mathrm{MASK}} -\log p_\theta(w_i \mid z)\Big]\, dt \;\ge\; -\log p_\theta(w).$$

Na difusão a rede define só condicionais. A verossimilhança do modelo é a média, sobre as ordens de revelação $\sigma$, do produto das condicionais em cada ordem, $p_\theta(w) = \mathbb{E}_\sigma\big[\prod_{i=1}^{L} p_\theta(w_{\sigma(i)} \mid w_{\sigma(<i)})\big]$, e a NELBO é a média de $-\log \prod$ sobre as mesmas ordens (Ou et al., 2024, Teorema 2). Pela desigualdade de Jensen, ela limita $-\log p_\theta(w)$ por cima.

**Bloco #, da integral à soma.** Seja $F_l = \mathbb{E}_{|S| = l}\big[\sum_{i \in S} -\log p_\theta(w_i \mid z)\big]$. Como $l \sim \mathrm{Bin}(L, t)$ e todos os termos são não negativos, o teorema de Tonelli permite trocar soma e integral,

$$\mathrm{NELBO} = \sum_{l=1}^{L} \binom{L}{l} \Big(\int_0^1 t^{l-1}(1-t)^{L-l}\, dt\Big) F_l = \sum_{l=1}^{L} \binom{L}{l} \mathrm{B}(l, L-l+1)\, F_l = \sum_{l=1}^{L} \frac{F_l}{l},$$

porque $\binom{L}{l}\mathrm{B}(l, L-l+1) = \frac{L!}{l!(L-l)!} \cdot \frac{(l-1)!(L-l)!}{L!} = \frac{1}{l}$.

**Bloco #, estimador.** Logo $\mathrm{NELBO} = \mathbb{E}_{l \sim U\{1..L\}}\big[\tfrac{L}{l} F_l\big]$, e o estimador para uma janela é $\hat\ell_{dif}(w; l, S) = \frac{L}{l}\sum_{i \in S} -\log p_\theta(w_i \mid z)$. No treino, a perda do lote é a média de $\hat\ell / L$. Código em `perda_difusao`, no `slm/perda.py`.
""", DIF), ("md", r"""
**Bloco #, otimização.** O gradiente $g = \nabla_\theta\, \text{perda}$, recortado em norma 1,0, alimenta o AdamW, com $m \leftarrow \beta_1 m + (1-\beta_1) g$, $v \leftarrow \beta_2 v + (1-\beta_2) g^2$ e $\theta \leftarrow \theta - \eta_t\big(\hat m / (\sqrt{\hat v} + \epsilon) + \lambda \theta\big)$, em que $\hat m$ e $\hat v$ são corrigidos de viés e o decaimento $\lambda$ só se aplica às matrizes de pesos, ficando de fora vieses, LayerNorm e posições. A taxa $\eta_t$ sobe linearmente até o pico $\eta$ no aquecimento e depois segue $\eta_t = \eta_{min} + \tfrac{1}{2}(\eta - \eta_{min})(1 + \cos \pi\, \text{progresso})$. Em fp16 a perda é multiplicada por uma escala $S$ antes da retropropagação, o gradiente é dividido por $S$ antes do passo, e um passo com valor infinito é pulado enquanto $S$ diminui. Código em `slm/treino.py`. Os parâmetros treinados voltam à caixa 5 e servem às caixas 7 e 8.
""", AMBOS), ("code", r"""
from slm.perda import perda_ar
w6 = torch.randint(4, tok.vocab_size, (1, 16))
with torch.no_grad():
    z6 = mini(torch.cat([torch.ones(1, 1, dtype=torch.long), w6[:, :-1]], 1), True)
    a_mao = (-torch.log_softmax(z6.double(), -1).gather(-1, w6[..., None])).sum()
    print("l_AR à mão:", round(float(a_mao), 4), "| pacote:", round(float(perda_ar(mini, w6)[1]), 4))
zz = torch.zeros(5, requires_grad=True)
F.cross_entropy(zz[None], torch.tensor([2])).backward()
print("gradiente da entropia cruzada = p - e_y:", torch.allclose(zz.grad, torch.softmax(zz.detach(), 0) - torch.eye(5)[2]))
""", AR), ("code", r"""
from scipy.special import beta, comb
from slm.mascaramento import mascarar
from slm.perda import perda_difusao
print("identidade C(L,l) B(l, L-l+1) = 1/l:", all(abs(comb(256, l) * beta(l, 257 - l) - 1 / l) < 1e-9 for l in (1, 2, 17, 128, 256)))
w6 = torch.randint(4, tok.vocab_size, (1, 16))
m6 = torch.zeros(1, 16, dtype=torch.bool)
m6[0, [2, 5, 11]] = True
with torch.no_grad():
    lp = torch.log_softmax(mini(mascarar(w6, m6), False).double(), -1).gather(-1, w6[..., None])[..., 0]
    a_mao = 16 / 3 * float(-lp[m6].sum())
    print("(L/l) soma das CE mascaradas à mão:", round(a_mao, 4), "| pacote:", round(float(perda_difusao(mini, w6, m6)[1]), 4))
zz = torch.zeros(5, requires_grad=True)
F.cross_entropy(zz[None], torch.tensor([2])).backward()
print("gradiente da entropia cruzada = p - e_y:", torch.allclose(zz.grad, torch.softmax(zz.detach(), 0) - torch.eye(5)[2]))
""", DIF)],
    7: [("md", r"""
### Blocos de construção

**Bloco #, Gumbel-max.** Se $g_v = -\log(-\log u_v)$ com $u_v \sim U(0,1)$ independentes, então $\arg\max_v (\log p_v + g_v)$ tem distribuição $p$. O sorteio vira um argmax, e por isso a precisão de $u$ importa.
""", AMBOS), ("md", r"""
**Bloco #, geração ancestral.** Começa-se com $x = (\text{BOS})$, sorteia-se $w_1 \sim p_\theta(\cdot \mid \text{BOS})$, acrescenta-se $w_1$ à entrada e repete-se, até o `[EOS]` ou o fim da janela. É a regra da cadeia percorrida no sentido da geração. Código em `sortear` e `gerar_ar`, no `slm/geracao.py`. As amostras seguem para a caixa 10.
""", AR), ("md", r"""
**Bloco #, cronograma de revelação.** No passo de $t = k/T$ para $s = (k-1)/T$, cada posição mascarada é revelada com probabilidade $\frac{\alpha_s - \alpha_t}{1 - \alpha_t} = \frac{t - s}{t} = \frac{1}{k}$, e no último passo ($k = 1$) todas são reveladas. Código em `sortear` e `gerar_difusao`, no `slm/geracao.py`. As amostras seguem para a caixa 10.
""", DIF), ("code", r"""
from slm.geracao import sortear
p7 = torch.tensor([0.5, 0.3, 0.15, 0.05])
s7 = sortear(torch.log(p7).expand(200000, 4), torch.Generator().manual_seed(0))
freq = (torch.bincount(s7, minlength=4) / len(s7)).tolist()
print("p alvo:", [round(x, 3) for x in p7.tolist()], "| frequências do Gumbel-max:", [round(x, 3) for x in freq])
""", AMBOS)],
    8: [("md", r"""
### Blocos de construção

**Bloco #, perda do documento.** $\mathrm{NLL}(d) = \sum_{w \in d} \ell_{AR}(w)$, somando as janelas da caixa 3. No modo autorregressivo esse valor é exato.

**Bloco #, bits por byte.** $\mathrm{bpb}(D) = \dfrac{\sum_{d \in D} \mathrm{NLL}(d)}{\ln 2 \cdot \sum_{d \in D} B_d}$. O $\ln 2$ converte nats em bits, e dividir por bytes em vez de tokens torna o número comparável entre tokenizadores.
""", AR), ("md", r"""
### Blocos de construção

**Bloco #, perda do documento.** $\mathrm{NLL}(d) = \sum_{w \in d} \hat\ell_{dif}(w)$, somando as janelas da caixa 3, cada uma com a média de $K$ estimativas com $l$ estratificado. Com $K = 1$, o valor é uma estimativa não enviesada de um limite superior da log-verossimilhança negativa, e carrega ruído de Monte Carlo.

**Bloco #, bits por byte.** $\mathrm{bpb}(D) = \dfrac{\sum_{d \in D} \mathrm{NLL}(d)}{\ln 2 \cdot \sum_{d \in D} B_d}$. O $\ln 2$ converte nats em bits, e dividir por bytes em vez de tokens torna o número comparável entre tokenizadores.
""", DIF), ("md", r"""
**Bloco #, temperatura.** Com $\beta = 1/T$, a perda média de validação é $\mathcal{L}(\beta) = \frac{1}{N}\sum_n \big(-\beta o_{n,y_n} + \log \sum_v e^{\beta o_{n,v}}\big)$. As derivadas são $\mathcal{L}'(\beta) = \frac{1}{N}\sum_n (\mathbb{E}_{p_\beta}[o_n] - o_{n,y_n})$ e $\mathcal{L}''(\beta) = \frac{1}{N}\sum_n \mathrm{Var}_{p_\beta}[o_n] \ge 0$, então $\mathcal{L}$ é convexa e todo ponto estacionário é mínimo global. Convexidade não basta para o método de Newton puro convergir, nem garante que o mínimo seja finito, já que, se todas as previsões estiverem certas, $\beta$ cresce sem limite. O pacote aplica o passo $\beta \leftarrow \beta - \mathcal{L}'/\mathcal{L}''$, encurta o passo pela metade enquanto ele levaria $\beta$ a zero ou abaixo, e para quando o passo fica menor que $10^{-10}$ ou depois de 100 iterações.

**Bloco #, ECE.** Com faixas de confiança $b$, $\mathrm{ECE} = \sum_b \frac{n_b}{N}\, |\bar c_b - \bar a_b|$. Código em `slm/avaliacao.py` e `slm/calibracao.py`. Os bits por byte de cada documento seguem para a caixa 9. A célula abaixo confere a agregação em bits por byte a partir dos nats por documento devolvidos pelo pacote.
""", AMBOS), ("code", r"""
import math
from slm.avaliacao import avaliar
tres = validacao[:3]
r8 = avaliar(modelo, tres, MODO, L, dispositivo=DISPOSITIVO)
a_mao = sum(x["nats"] for x in r8["documentos"]) / (math.log(2) * sum(d["bytes"] for d in tres))
print("bpb à mão:", round(a_mao, 5), "| pacote:", round(r8["bpb"], 5))
""", AMBOS)],
    9: [("md", r"""
### Blocos de construção

**Bloco #, bootstrap por obra.** Cada réplica reamostra as obras com reposição, recalcula os bits por byte como razão de somas e guarda o valor. O intervalo vai do percentil 2,5% ao 97,5% das réplicas. Reamostrar obras inteiras respeita a dependência entre capítulos da mesma obra.

**Bloco #, sementes.** Uma única corrida não diz quanto o resultado varia com a inicialização e com a ordem dos dados. O protocolo do projeto treina várias sementes e reamostra em dois níveis, obras e sementes. Com $\hat\sigma$ estimado em $m$ sementes piloto, o limite superior é $\sigma_{sup} = \hat\sigma\sqrt{(m-1)/\chi^2_{0,025;\,m-1}}$, e o número de sementes é o menor $n$ com $n \ge (t_{0,975;n-1} + t_{0,8;n-1})^2 (\sigma_{sup}/\Delta_{min})^2$.

**Bloco #, comparação entre os modos.** A semente $s$ fixa a inicialização e a ordem dos dados nos dois modos, e a diferença pareada é $\Delta_s = \mathrm{bpb}_{dif,s} - \mathrm{bpb}_{AR,s}$. Se o intervalo de $\Delta$ fica abaixo de zero, a difusão é melhor. Se fica acima, o modo autorregressivo supera o limite da difusão, sem conclusão sobre a verossimilhança verdadeira dela. Se contém o zero, o resultado é inconclusivo. Essa leitura precisa das corridas dos dois notebooks, e o resultado do protocolo completo fica em `docs/`, no repositório. Código em `slm/estatistica.py`.
""", AMBOS), ("code", r"""
from slm.estatistica import fator_sigma_sup, numero_sementes
print("fator de sigma_sup com 10 sementes:", round(fator_sigma_sup(10), 4))
for r in (0.5, 0.75, 1.0):
    print(f"sigma/Delta_min = {r}: n =", numero_sementes(r, 1.0, m=10).n)
""", AMBOS)],
    10: [("md", r"""
### Blocos de construção

**Bloco #, taxa de léxico.** $\rho(a) = \frac{|\{x \in \text{palavras}(a) : x \in \Lambda_{treino}\}|}{|\text{palavras}(a)|}$, com $\Lambda_{treino}$ o léxico da partição de treino.

**Bloco #, maior trecho copiado.** $c(a) = \max\{k : \text{algum trecho de } k \text{ palavras de } a \text{ ocorre no texto de treino}\}$. A busca usa hash de $k$-gramas e busca binária em $k$, porque se um trecho de $k$ palavras ocorre, todos os seus subtrechos também ocorrem, e a propriedade é monótona em $k$.

**Bloco #, curva de deleção.** Com a decisão $\hat y$ e $p_0 = p(\hat y \mid \text{contexto completo})$, a importância da palavra $j$ é $p_0 - p(\hat y \mid \text{contexto sem } j)$. Remover as $\mathrm{round}(f n)$ palavras mais importantes dá $C(f) = \min\big(p(\hat y \mid \text{contexto restante})/p_0, 1\big)$, e $\mathrm{AUC} = \int_0^1 C(f)\, df$ pela regra do trapézio. $C(1)$ mede o viés intrínseco.

**Bloco #, Spearman.** $\rho_S$ é a correlação de Pearson entre os postos de $p_0$ e os postos da AUC, item a item. Código em `slm/leitura.py` e `slm/delecao.py`.
""", AMBOS), ("code", r"""
import numpy as np
from scipy.stats import rankdata, spearmanr
x10, y10 = np.array([0.2, 0.5, 0.1, 0.9]), np.array([0.7, 0.4, 0.8, 0.3])
print("Spearman à mão:", round(float(np.corrcoef(rankdata(x10), rankdata(y10))[0, 1]), 4),
      "| scipy:", round(float(spearmanr(x10, y10).statistic), 4))
grade = np.linspace(0, 1, 11)
C10 = np.array([1, .6, .5, .45, .4, .38, .36, .35, .34, .33, .33])
print("AUC pela regra do trapézio:", round(float(np.trapezoid(C10, grade)), 4))
""", AMBOS)],
}

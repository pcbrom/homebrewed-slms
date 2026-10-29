# Pesquisa: difusão discreta para texto

Levantamento de 22/10/2026. Equações e números conferidos nas versões do arXiv indicadas em [referencias/REFERENCIAS.md](../referencias/REFERENCIAS.md). A numeração de equações muda entre versões de um mesmo artigo, e por isso cada citação indica a seção. Cálculos feitos por este projeto, e não tomados das fontes, vêm marcados como tal.

## 1. Notação

$x$ é a sequência limpa de comprimento $L$, $x^\ell$ o token na posição $\ell$, $m$ o token de máscara e $z_t$ a sequência mascarada no tempo $t \in [0,1]$. O cronograma $\alpha_t$ é a probabilidade de um token permanecer intacto, decrescente, com $\alpha_0 \approx 1$ e $\alpha_1 \approx 0$, de modo que $\alpha'_t < 0$.

## 2. Formulações

A literatura recente converge para a difusão mascarada, também chamada absorvente: cada token, de forma independente, é trocado pela máscara com probabilidade $1 - \alpha_t$, e uma vez mascarado permanece mascarado. As formulações diferem menos no processo direto do que na parametrização da rede, na forma da perda e na estratégia de amostragem.

| Formulação | Processo direto | O que a rede prevê | Recebe $t$? | Amostragem |
|---|---|---|---|---|
| D3PM (Austin et al., 2021) | Cadeia discreta em $T$ passos, $q(x_t \mid x_{t-1}) = \mathrm{Cat}(x_{t-1} Q_t)$, com matrizes uniforme, absorvente e outras | $\tilde p_\theta(\tilde x_0 \mid x_t)$, combinado com o posterior do processo | Não explícito na notação | Ancestral em $T$ passos |
| SEDD (Lou, Meng e Ermon, 2023) | Cadeia de Markov em tempo contínuo, $Q_t = \sigma(t) Q$ | Escore concreto $s_\theta(x,t)_y \approx p_t(y)/p_t(x)$ | Sim, pelo ruído total, via adaLN-zero | Euler ou $\tau$-leaping |
| MDLM (Sahoo et al., 2024) | $q(z_t \mid x) = \mathrm{Cat}(\alpha_t x + (1-\alpha_t) m)$ | Distribuição sobre o token limpo, com logit da máscara em $-\infty$ e cópia dos tokens não mascarados | Opcional, sem perda relevante de perplexidade | Ancestral, com cache e semi-autoregressiva |
| MD4 (Shi et al., 2024) | $\bar Q(t) = \alpha_t I + (1-\alpha_t) \mathbf{1} e_m^\top$ | $\mu_\theta(x_t, t)$ | Opcional | Ancestral em grade cosseno |
| RADD (Ou et al., 2024) | Absorvente | Condicional dos dados limpos $c_\theta(x_t)$ | Não | Com cache |
| LLaDA (Nie et al., 2025) | Máscara independente com probabilidade $t \sim U[0,1]$ | $p_\theta(x_0^i \mid x_t)$ | Não | Remascaramento por baixa confiança e semi-autoregressiva por blocos |
| BD3-LM (Arriola et al., 2025) | Absorvente dentro de cada bloco | $p_\theta(x^b \mid x^b_t, x^{<b})$ | Não | Bloco a bloco com cache de chaves e valores |

O resultado de Ou et al. (2024, Teorema 1) explica por que a entrada de tempo pode ser dispensada. No caso absorvente, o escore concreto fatora num escalar que depende só do tempo vezes a condicional dos dados limpos dado o contexto não mascarado. A fração de máscaras na entrada já carrega a informação que o tempo daria. A Tabela 12 do MDLM confirma isso na prática, com perplexidades semelhantes com e sem condicionamento no tempo. Para este projeto, a consequência é direta: sem entrada de tempo, a rede do modo de difusão e a do modo autoregressivo têm exatamente os mesmos parâmetros, e a diferença entre os dois modos se reduz à máscara de atenção e à perda.

O BD3-LM fica entre os dois extremos. A atenção é causal entre blocos e bidirecional dentro de cada bloco, e com bloco de tamanho 1 a formulação coincide com o modelo autoregressivo. Ele oferece um terceiro modo natural para a bancada, sem rede nova.

## 3. A perda

A forma geral da perda em tempo contínuo é a mesma no MDLM (seção 3.4) e no MD4 (seção 3):

$$\mathcal{L} = \mathbb{E}_{q}\int_0^1 \frac{\alpha'_t}{1-\alpha_t} \sum_{\ell} \log \langle x_\theta^\ell(z_t), x^\ell \rangle \, dt,$$

em que só as posições mascaradas contribuem. Como $\alpha'_t < 0$ e o logaritmo de uma probabilidade é negativo, o integrando é positivo. Com o cronograma linear $\alpha_t = 1 - t$, o peso vale $\alpha'_t / (1 - \alpha_t) = -1/t$, e a perda se torna

$$\mathcal{L} = \mathbb{E}_{t \sim U(0,1]}\, \mathbb{E}_{z_t} \left[ \frac{1}{t} \sum_{\ell:\, z_t^\ell = m} -\log p_\theta(x^\ell \mid z_t) \right],$$

forma que aparece na Tabela 4 do MD4, na Eq. 3 do LLaDA e no trabalho de Nie et al. (2024). O MDLM mostra ainda que, escrita em função de $\gamma = \log(1 - \alpha_t)$, a perda não depende da forma do cronograma (seção 3.4).

A perda é um limite superior da log-verossimilhança negativa, $-\log p_\theta(x) \le \mathcal{L}$, e portanto $\mathrm{PPL} \le \exp(\mathcal{L}/L)$. A legenda da Tabela 3 do MDLM registra que "all perplexities for diffusion models are upper bounds". Em tempo discreto com $T$ passos o limite afrouxa, como mostra a Tabela 11 do MDLM no OpenWebText: 23,05 com $T = \infty$, 23,15 com $T = 1000$, 24,35 com $T = 100$ e 42,18 com $T = 10$.

A mesma forma de perda sem o peso $1/t$, isto é, a entropia cruzada média sobre os tokens mascarados, aparece em implementações didáticas. Ela treina um modelo que gera texto, mas não é a NELBO e não fornece limite de perplexidade, o que impede a comparação com o modo autoregressivo em bits por byte.

## 4. Estimadores e variância

O peso $1/t$ amplifica a variância quando $t$ é pequeno. As fontes usam três recursos:

- Amostragem de $t$ com baixa discrepância. O MDLM sorteia $t_i \sim U[\frac{i-1}{N}, \frac{i}{N}]$ dentro do lote (suplemento D.3), e o código usa $\varepsilon = 10^{-3}$ como limite inferior de $t$, o que limita o peso a 1.000. O MD4 usa o truque antitético de Kingma et al. (2021) e $\varepsilon = 10^{-4}$.
- Estimador por número de máscaras. O LLaDA sorteia $l \sim U\{1, \dots, L\}$, mascara exatamente $l$ tokens sem reposição e pondera por $L/l$, o que estabiliza a estimativa com 128 amostras de Monte Carlo, contra mais de 1.000 da forma por $t$ (apêndice A.2).
- Taxa de máscara recortada. O BD3-LM sorteia a taxa em $U[\beta, \omega]$ e escolhe o intervalo por grade minimizando a variância. Com bloco 4, $U[0{,}45, 0{,}95]$ dá perplexidade 29,21 e variância 6,24, contra 30,18 e 23,45 com $U[0,1]$ (Tabela 8).

Como em média metade dos tokens fica mascarada, cada passo da difusão produz gradiente a partir de cerca de metade das posições, enquanto o autoregressivo usa todas. Esse fator entra na discussão de orçamento do [protocolo](pesquisa_protocolo.md).

## 5. Amostragem

A geração parte de uma sequência toda mascarada, ou de um prefixo seguido de máscaras, e revela tokens em $K$ passos. Duas escolhas governam a qualidade:

- **Quais tokens revelar.** No MDLM, cada máscara se revela com probabilidade $(\alpha_s - \alpha_t)/(1 - \alpha_t)$ no passo de $t$ para $s$, de forma aleatória. O LLaDA revela primeiro os tokens de maior confiança e remascara os demais. Na Tabela 9 do LLaDA (8B, 256 tokens, 256 passos), a troca do aleatório pela baixa confiança leva GSM8K de 21,3 para 70,0 e HumanEval de 11,6 para 32,9.
- **Blocos.** A geração semi-autoregressiva por blocos do LLaDA leva GSM8K de 69,4 para 77,5 com blocos de 32 tokens (Tabela 8). O artigo atribui parte do ganho ao término precoce que o preenchimento com EOS provoca na geração não blocada.

O número de passos troca qualidade por velocidade. No D3PM absorvente, a perplexidade no LM1B vai de 76,9 com 1.000 passos para 83,6 com 64. Quanto à velocidade contra o autoregressivo, os relatos são de ganho modesto sem otimização específica: 1,5 a 1,8 vez no LLaDA contra um LLaMA 3 8B com cache, em lote 1 (apêndice B.7). Ganhos maiores, como os de Wu et al. (2025), comparam o LLaDA com ele mesmo após cache e decodificação paralela, e não com um autoregressivo.

## 6. Precisão numérica na amostragem

Zheng et al. (2024) mostraram que a amostragem categórica por Gumbel, $g = -\log(-\log u)$, fica truncada pela precisão de $u$, o que equivale a baixar a temperatura. O valor máximo do ruído é cerca de 16,64 em fp32 e 36,74 em fp64 (Tabela 1). Com 2.048 passos, a perplexidade gerativa cai de 126,11 em 64 bits para 31,24 em 32 bits, com a entropia caindo de 5,66 para 5,17 (Tabela 2). A aparente qualidade em 32 bits é, portanto, perda de diversidade. Com a precisão corrigida, a perplexidade gerativa das difusões mascaradas fica em torno de 100, contra menos de 40 dos autoregressivos (seção 6.1).

O artigo não trata fp16, a precisão de treino na T4. A fórmula $g_{max} = -\log(-\log(1 - 2^{-p}))$, com $p$ bits de mantissa, reproduz os valores de Zheng para fp32 ($p = 24$) e fp64 ($p = 53$), e dá 7,62 em fp16 ($p = 11$) e 5,54 em bf16 ($p = 8$). Uma medida em CPU com 20 milhões de sorteios de `torch.rand` em fp16 confirmou 2.048 valores distintos e $g_{max} = 7{,}6244$. Esses dois resultados são cálculo e medida deste projeto. Deles decorre que um token com probabilidade abaixo de cerca de $6{,}4 \times 10^{-5}$ vezes a do mais provável nunca é sorteado se o ruído for gerado em fp16, o que agrava o efeito descrito por Zheng. O comportamento em CUDA e o efeito sobre a perplexidade gerativa não foram medidos.

O código de referência mostra os dois caminhos. A implementação oficial do MDLM gera o ruído com `torch.rand_like` sobre as probabilidades e herda o tipo delas, enquanto LLaDA e BD3-LM convertem para float64 antes do sorteio, citando Zheng et al. A amostragem deste projeto deve ocorrer em float64, independentemente da precisão do treino.

## 7. Resultados publicados contra o autoregressivo

Com o mesmo número de tokens de treino, a difusão mascarada fica atrás do autoregressivo em perplexidade, e o valor dela é um limite superior. A diferença relativa abaixo foi calculada por este projeto como $\mathrm{PPL}_{dif}/\mathrm{PPL}_{AR} - 1$.

| Fonte | Dados e escala | AR | Difusão (limite superior) | Diferença |
|---|---|---|---|---|
| MDLM, Tabela 1 | LM1B, 110M, 33B tokens | 22,32 | 27,04 | 21,1% |
| MDLM, Tabela 1 | LM1B, 327B tokens | 20,86 | 23,00 | 10,3% |
| MDLM, Tabela 2 | OpenWebText, 262B tokens | 17,54 | 23,21 | 32,3% |
| BD3-LM, Tabela 4 | OpenWebText, 524B tokens, bloco 4 | 17,54 | 20,73 | 18,2% |
| SEDD, Tabela 3 | LM1B | 31,98 | 32,79 | 2,5% |

No regime de uma época, Nie et al. (2024) estimam que a difusão mascarada precisa de "approximately 16 times more computational resources" para igualar o autoregressivo, com configurações a partir de 19M de parâmetros sem embedding. Prabhudesai et al. (2025) mostram a inversão quando os dados são escassos e repetidos: com 100M de tokens únicos do C4, a perda passa de 10,65 contra 7,07 em uma época para 3,55 contra 3,71 com muitas épocas, a favor da difusão, e a meia-vida de reuso dos dados é de cerca de 500 épocas na difusão contra cerca de 31 no autoregressivo. Ni et al. (2025) relatam resultado na mesma direção. O artigo de Prabhudesai traz inconsistências internas entre o texto e a Tabela 1 e na unidade de uma constante da lei de potência, registradas como incerteza.

Nenhuma comparação revisada por pares, pareada em perplexidade, foi encontrada na faixa de 10M a 30M de parâmetros. Para este projeto, que treina por muitas épocas num corpus pequeno (ver [pesquisa de corpus](pesquisa_corpus.md)), o regime relevante é o de Prabhudesai, e se o cruzamento se repete nessa escala é pergunta que só a medida responde.

## 8. Implementações de referência

| Repositório | Licença | Observação |
|---|---|---|
| kuleshov-group/mdlm | Apache-2.0 | Referência do MDLM |
| kuleshov-group/bd3lms | Apache-2.0 | Difusão por blocos |
| louaaron/Score-Entropy-Discrete-Diffusion | MIT | SEDD |
| google-deepmind/md4 | Apache-2.0 | Em JAX |
| ML-GSAI/RADD | Apache-2.0 | RADD |
| ML-GSAI/SMDM | Apache-2.0 | Código de Nie et al. (2024) |
| ML-GSAI/LLaDA | Sem licença detectada | Uso como leitura, não como base de código |
| karpathy/nanoGPT | MIT | Base autoregressiva |

As licenças foram lidas pela API do GitHub em 22/10/2026.

## 9. Consequências para as decisões

As escolhas finais estão na [decisão de formulação](decisao_formulacao.md).

- A formulação candidata é a difusão mascarada com cronograma linear e perda ponderada por $1/t$, sem entrada de tempo, o que mantém a mesma rede nos dois modos.
- A estimativa da perda precisa de recurso de redução de variância, a escolher entre baixa discrepância e o estimador por número de máscaras.
- A amostragem ocorre em float64, com a entropia relatada junto da perplexidade gerativa.
- A difusão por blocos é um terceiro modo possível sem rede nova.
- O regime de épocas da comparação é definido na [decisão de orçamento](decisao_orcamento.md).

Referências completas, versões e licenças em [referencias/REFERENCIAS.md](../referencias/REFERENCIAS.md).

# Decisão: formulação do modo de difusão

**Estado:** aceita pelo autor em 22/10/2026. **Data:** 22/10/2026. **Veredito do debate adversarial:** condicional, sem objeção fatal remanescente, convergido na rodada 3.

## Questão

Que formulação de difusão discreta usar no SLM, na mesma rede Transformer do modo autoregressivo, de forma que a perda seja um limite superior da log-verossimilhança negativa e permita comparação em bits por byte.

## Decisão

**Processo direto.** Máscara absorvente com cronograma linear, $\alpha_t = 1 - t$.

**Rede.** A mesma do modo autoregressivo, sem entrada de tempo, com atenção bidirecional. O logit do token `[MASK]` fica em $-\infty$ nos dois modos, e na difusão os tokens não mascarados são copiados na saída, como na parametrização SUBS do MDLM.

**Perda de treino.** Forma por número de máscaras: sorteia-se $l \sim U\{1, \dots, L\}$, estratificado no lote, mascaram-se exatamente $l$ posições sem reposição e a perda da sequência é

$$\frac{L}{l} \cdot \frac{1}{L} \sum_{i \in S} -\log p_\theta(x^i \mid x_{\setminus S}) = \frac{1}{l} \sum_{i \in S} -\log p_\theta(x^i \mid x_{\setminus S}),$$

com média sobre o lote. Essa forma é a NELBO exata, sem limite inferior de $t$ (Ou et al., 2024, Teorema 2 e apêndice C, e Nie et al., 2025, Eq. 14).

**Avaliação.** Pela mesma forma, inteira em fp32, com $\mathrm{bpb} = \sum \text{nats} / (\ln 2 \cdot \text{bytes})$.

**Objeto modelado.** Janelas de exatamente $L$ tokens, alinhadas ao início do documento, com EOS no fim e preenchidas com EOS até $L$. Os dois modos preveem todas as posições, o preenchimento incluído, com os nats contados e zero bytes. Treino e avaliação usam o mesmo janelamento, sem concatenar documentos.

**Número de amostras na avaliação.** A variância da média é $(\sigma_\Delta^2 + \sigma_w^2 / K) / n$. Com custo $nK = C$ fixo, ela cresce com $K$. Todas as janelas da validação entram com $K = 1$, e só a sobra do orçamento de avaliação vira $K = \lfloor C / N_w \rfloor$.

**Amostragem.** Ancestral, com revelação aleatória e sorteio em float64, no número de passos declarado antes da comparação lado a lado. A revelação por baixa confiança e a geração semi-autoregressiva por blocos entram só como variantes rotuladas, fora do resultado confirmatório.

**Critério de leitura**, declarado antes da corrida, com $\Delta = \mathrm{bpb}_{dif} - \mathrm{bpb}_{AR}$ pareado:

- Intervalo abaixo de zero: a difusão é melhor em verossimilhança, resultado conclusivo.
- Intervalo acima de zero: o autoregressivo supera o limite da difusão, sem conclusão sobre a verossimilhança verdadeira da difusão.
- Intervalo contendo zero: inconclusivo.

## Justificativa

**A forma por número de máscaras é exata.** Dado $t$, o número de máscaras segue $l \sim \mathrm{Bin}(L, t)$, e dado $l$ o conjunto mascarado é uniforme. Como os termos são não negativos, o teorema de Tonelli permite trocar soma e integral, e

$$\int_0^1 \frac{1}{t} \sum_{l \ge 1} \binom{L}{l} t^l (1-t)^{L-l} F_l \, dt = \sum_l \binom{L}{l} B(l, L-l+1) F_l = \sum_l \frac{F_l}{l},$$

que é $\mathbb{E}_{l \sim U\{1..L\}}[(L/l) F_l]$. A identidade foi conferida em aritmética racional para $L \le 64$ e para todo $l$ de 1 a 256 com $L = 256$.

**Ela domina a forma por $t$.** Dado $l$, $t \sim \mathrm{Beta}(l+1, L-l+1)$ e $\mathbb{E}[1/t \mid l] = (L+1)/l$. Pelo teorema de Rao-Blackwell, a forma por número de máscaras tem a mesma média e segundo momento menor, $L \sum_l \mathbb{E}[f_l^2]/l^2$ contra $(L+1) \sum_l \mathbb{E}[f_l^2]/l^2$, na perda e em cada coordenada do gradiente. Ela também elimina o sorteio inútil de $l = 0$ e o corte em $t \ge \varepsilon$. Esse corte remove do limite uma massa igual a $L\varepsilon$, com viés relativo da ordem de $\varepsilon$, e sem ele a forma por $t$ tem variância infinita.

**O preenchimento fecha um vazamento.** Se a difusão recebesse o comprimento do documento sem pagar por ele, ganharia até $\log_2 L$ bits por documento contra o autoregressivo, que paga pelo EOS. Num teste com condicionais exatas em documentos sintéticos, a diferença foi de 1,4855 bits por documento, igual à entropia do comprimento, e caiu a $-2{,}1 \times 10^{-17}$ com o preenchimento previsto.

**A amostragem aleatória preserva o sorteio.** A revelação por baixa confiança usa argmax e anula o sorteio em float64, e o próprio LLaDA recomenda remascaramento "purely random" como padrão.

## Portões, em ordem

| Portão | Onde | Critério | Se falhar |
|---|---|---|---|
| G0 | CPU, código do pacote | Com $L \le 6$ e vocabulário pequeno, enumerando as $L!$ ordens: forma por número de máscaras igual à média sobre ordens, normalização $|\Sigma_x \mathbb{E}_\sigma[\prod p] - 1| \le 10^{-9}$ e NELBO $\ge -\log \mathbb{E}_\sigma[\prod p]$. Os defeitos injetados de logit de `[MASK]` mantido e de alvo visível na própria previsão precisam reprovar | Bloqueia tudo |
| G1 | CPU, código do pacote | Teste de comprimento com o iterador e o preenchimento do pacote, diferença até $10^{-9}$, e soma de bytes do iterador igual à da validação | Bloqueia tudo |
| G2 | Corpus | Fração de preenchimento medida antes de fixar $L$ | $L$ revisto |
| G3 | Orçamento | $C \ge N_w$ | Sobe $C$, não se cortam documentos |
| G4 | Pré-registro | Critério de leitura registrado antes da corrida | Nada roda |
| G5 | Amostras | Entropia em float64 contra fp32 relatada | |

## Riscos declarados

- A folga entre a NELBO e a verossimilhança verdadeira não se mede na escala real. Por isso o resultado com intervalo acima de zero é estruturalmente inconclusivo sobre a verossimilhança da difusão.
- O teste de comprimento com condicionais exatas é igual por construção. Ele só vale como teste quando percorre o código do pacote (G1).
- Os portões G0 e G1 usam escala de brinquedo e não detectam defeitos que só aparecem em escala real, como máscara de atenção dependente de comprimento ou estouro em fp16, cobertos pela [decisão de precisão](decisao_precisao.md).
- O preenchimento consome orçamento, tanto mais quanto mais documentos curtos houver (G2).

## Fora do escopo

- Difusão por blocos (BD3-LM), cujo cronograma recortado não é NELBO e cujo ganho relatado é instável.
- Revelação por baixa confiança como amostrador principal.
- Comparação entre os estimadores estratificados por $t$ e por número de máscaras, irrelevante para a validade do limite.

## Registro da medida

Portão G2, em 23/10/2026, com o tokenizador de $V = 4.096$ no corpus A: a fração de preenchimento do treino é de 6,3% com $L = 128$, 12,2% com $L = 256$ e 22,5% com $L = 512$. Fica $L = 256$, com 2.780 janelas de treino, 657 de validação e 579 de teste.

Portões G0 e G1 do pacote, em `tests/test_portoes_formulacao.py`, com vocabulário mínimo e $L = 4$, em float64. No G0, a rede correta soma 1,0 sobre todas as sequências, a rede com o logit de `[MASK]` livre soma 0,5906 e a rede que vê o próprio alvo soma 1,2837. No G1, a distribuição sintética do teste tem entropia do comprimento de 0,8559 nat, ou 1,2347 bit, e a vantagem indevida da difusão que recebe o comprimento coincide com ela dentro de $10^{-9}$. O valor de 1,4855 bit citado na justificativa vem de outra distribuição sintética, usada antes de o teste ser incorporado ao pacote. A medida reproduzível é a do teste.

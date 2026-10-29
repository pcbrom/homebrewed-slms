# Especificação

Este documento fixa as definições formais que o código implementa. A numeração das seções segue as caixas do [pipeline](ARQUITETURA.md). Valores marcados como convenção não têm fonte e são revistos antes do pré-registro, que os congela com hash.

## 1. Notação

Um documento $d$ é uma unidade do corpus (capítulo, entrada de diário ou lei), com texto normalizado em NFC e $B_d$ bytes em UTF-8. O vocabulário tem $V$ tokens, com os ids reservados `[PAD]` = 0, `[BOS]` = 1, `[EOS]` = 2 e `[MASK]` = 3. $L$ é o comprimento da janela e $B$ o tamanho do lote. Logaritmos são naturais, e as perdas estão em nats salvo indicação.

## 2. Tokenizador (caixa 2)

A codificação é $\tau(d) = (t_1, \dots, t_n)$, com $t_i \in \{4, \dots, V-1\}$. O contrato e os testes U1 a U6 estão na [decisão do tokenizador](decisao_tokenizador.md). Os bytes da métrica são sempre $B_d$, contados sobre o texto NFC do corpus e nunca sobre o texto reconstruído pelo tokenizador.

## 3. Janelamento (caixa 3)

A sequência do documento recebe um EOS no fim, $s(d) = (t_1, \dots, t_n, \text{EOS})$, de comprimento $n + 1$. Ela é cortada em $N_w(d) = \lceil (n+1)/L \rceil$ janelas consecutivas de $L$ tokens, alinhadas ao início, e a última é preenchida com EOS até $L$. Os tokens de preenchimento são previstos e contados em nats nos dois modos, com zero bytes. Não há contexto entre janelas, e documentos nunca são concatenados.

A fração de preenchimento do corpus é $\phi = \sum_d (L \cdot N_w(d) - n_d - 1) / \sum_d L \cdot N_w(d)$, medida antes de fixar $L$.

## 4. Modo autoregressivo (caixas 4b e 5)

Para a janela $w = (w_1, \dots, w_L)$, a entrada da rede é $(\text{BOS}, w_1, \dots, w_{L-1})$, com máscara de atenção causal, e a perda da janela é

$$\ell_{AR}(w) = \sum_{i=1}^{L} -\log p_\theta(w_i \mid \text{BOS}, w_{<i}).$$

No treino, a perda do lote é $\frac{1}{BL} \sum_{b=1}^{B} \ell_{AR}(w_b)$.

## 5. Modo de difusão (caixas 4a e 5)

**Mascaramento.** Dado $l \in \{1, \dots, L\}$, sorteia-se $S \subset \{1, \dots, L\}$ com $|S| = l$, uniforme entre os $\binom{L}{l}$ subconjuntos, e $z$ é $w$ com as posições de $S$ trocadas por `[MASK]`. A entrada da rede é $z$, sem token BOS adicional, com atenção bidirecional.

**Parametrização.** O logit de `[MASK]` vale $-\infty$ em todas as posições, nos dois modos. Na difusão, a saída das posições fora de $S$ é a cópia do token visível e não entra na perda.

**Estimador da janela.**

$$\hat\ell_{dif}(w; l, S) = \frac{L}{l} \sum_{i \in S} -\log p_\theta(w_i \mid z), \qquad \mathbb{E}_{l \sim U\{1..L\},\, S}\big[\hat\ell_{dif}\big] = \mathrm{NELBO}(w) \ge -\log p_\theta(w).$$

A igualdade com a NELBO em tempo contínuo e cronograma linear é o Teorema 2 de Ou et al. (2024), demonstrada na [decisão de formulação](decisao_formulacao.md).

**Sorteio estratificado no lote.** Com $U \sim U(0,1)$ comum ao lote e uma permutação aleatória $\pi$ de $\{1, \dots, B\}$, $u_b = (\pi(b) - 1 + U)/B$ e $l_b = 1 + \lfloor u_b L \rfloor$. Como a média de uniformes sobre os $B$ estratos é uniforme em $[0,1)$, a média do lote continua não viesada para a NELBO.

**Perda de treino.** $\frac{1}{BL} \sum_b \hat\ell_{dif}(w_b; l_b, S_b) = \frac{1}{B} \sum_b \frac{1}{l_b} \sum_{i \in S_b} -\log p_\theta(w_{b,i} \mid z_b)$, calculada em fp32. A normalização pelo número de tokens mascarados é proibida.

## 6. Treino e orçamento (caixa 6)

O custo de treino por token é o da seção 1 de [DIMENSIONAMENTO.md](DIMENSIONAMENTO.md),

$$C_{train} \approx 6N + 6\, n_{layer}\, L\, d + 6\, d\, V, \qquad N = 12\, n_{layer}\, d^2,$$

igual nos dois modos. Cada passo processa $BL$ tokens. Só contam os passos aplicados, isto é, aqueles em que o `GradScaler` não detectou estouro. Um passo pulado repete o mesmo lote da sequência pré-sorteada. O número de épocas de uma corrida é $E = P_{apl} \cdot B L / T_{treino}$, com $P_{apl}$ os passos aplicados e $T_{treino}$ o total de tokens das janelas de treino, preenchimento incluído. A taxa de aprendizado segue cosseno fechado na duração da própria corrida.

## 7. Geração (caixa 7)

**Sorteio.** Os logits saem em fp32, fora do autocast, e são convertidos para float64. O token é $\arg\max_v (\log p_v + g_v)$, com $g_v = -\log(-\log u_v)$ e $u_v \sim U(0,1)$ em float64. A temperatura é 1.

**Difusão.** A sequência começa com $L$ posições em `[MASK]`, ou com um prefixo fixo seguido de máscaras, e é revelada em $T_{gen}$ passos. No passo de $t = k/T_{gen}$ para $s = (k-1)/T_{gen}$, cada posição ainda mascarada é revelada com probabilidade $(t - s)/t$, e no último passo todas são reveladas. A posição revelada recebe o token sorteado da distribuição da rede, e as já reveladas não mudam. $T_{gen}$ é declarado antes da comparação lado a lado.

**Autoregressivo.** Sorteio token a token a partir de BOS, ou do prefixo, até $L$ tokens ou EOS.

## 8. Avaliação (caixa 8)

**Log-verossimilhança por documento.** $\mathrm{NLL}_{AR}(d) = \sum_{w \in d} \ell_{AR}(w)$. Para a difusão, $\widehat{\mathrm{NLL}}_{dif}(d) = \sum_{w \in d} \frac{1}{K} \sum_{k=1}^{K} \hat\ell_{dif}(w; l_k, S_k)$, com os $l_k$ estratificados sobre $K$ como na seção 5 e sementes de avaliação fixas.

**Bits por byte.** Para um conjunto $D$ de documentos,

$$\mathrm{bpb}(D) = \frac{\sum_{d \in D} \mathrm{NLL}(d)}{\ln 2 \cdot \sum_{d \in D} B_d}.$$

**Número de amostras.** Com orçamento de avaliação $C$ em passes de rede e $N_w$ janelas na validação, $K = \max(1, \lfloor C / N_w \rfloor)$. Se $C < N_w$, o orçamento sobe e nenhum documento é cortado.

**Bits por byte sem nomes.** Versão que exclui do numerador os nats das posições dentro dos trechos de nome próprio e do denominador os bytes desses trechos, com a lista de nomes extraída só do treino.

## 9. Comparação (caixa 9)

**Pareamento.** A semente $s$ fixa a inicialização e a ordem dos dados nos dois modos. Para cada semente, $\Delta_s = \mathrm{bpb}_{dif,s}(D_{teste}) - \mathrm{bpb}_{AR,s}(D_{teste})$.

**Intervalo.** Bootstrap em dois níveis, com $R = 10^4$ réplicas. Em cada réplica, reamostram-se com reposição as obras (grupos) e, de forma independente, as sementes, e calcula-se a média sobre as sementes reamostradas de $\mathrm{bpb}_{dif} - \mathrm{bpb}_{AR}$, cada bpb como razão de somas sobre os capítulos das obras reamostradas. O intervalo é o percentil de 2,5% a 97,5%.

**Cobertura.** No piloto, geram-se 1.000 conjuntos nulos multiplicando a diferença pareada de cada semente por um sinal $\pm 1$ sorteado. A cobertura é a fração desses conjuntos cujo intervalo contém zero, e precisa ser de pelo menos 0,93.

**Número de sementes.** Com $\hat\sigma$ o desvio de $\Delta_s$ entre as $m = 10$ sementes do piloto, $\sigma_{sup} = \hat\sigma \sqrt{(m-1)/\chi^2_{0{,}025;\, m-1}}$, que vale cerca de $1{,}83\,\hat\sigma$. O número de sementes é o menor $n$ com

$$n \ge \big(t_{0{,}975;\, n-1} + t_{0{,}8;\, n-1}\big)^2 \left(\frac{\sigma_{sup}}{\Delta_{min}}\right)^2,$$

com teto de 30 por modo.

**Margem mínima.** $\Delta_{min} = \frac{1}{4}\big(\mathrm{bpb}_{AR}^{(4)} - \mathrm{bpb}_{AR}^{(16)}\big)$, com o sobrescrito indicando as épocas, na melhor configuração da busca de hiperparâmetros e média de 2 sementes, medido na validação e congelado com hash antes do piloto.

**Leitura do resultado.** Intervalo abaixo de zero: difusão melhor em verossimilhança. Acima de zero: o autoregressivo supera o limite da difusão, sem conclusão sobre a verossimilhança verdadeira dela. Contendo zero: inconclusivo.

**Reprodução na T4.** TOST pareado por semente entre T4 e 5090 no ponto de 16 épocas, com margem $\Delta_{min}$ e dois testes $t$ unilaterais a $\alpha = 0{,}05$. A equivalência é declarada se os dois rejeitarem.

## 10. Leitura e coerência (caixa 10)

**Palavras.** Sequências máximas de letras Unicode, em minúsculas, extraídas do texto decodificado.

**Nível N1.** A taxa de uma amostra é a fração das palavras dela presentes no léxico da partição de treino. N1 é atingido num ponto da grade quando a taxa média das amostras do ponto fica a no máximo 5 pontos percentuais, em diferença absoluta, da taxa média de excertos humanos de teste de mesmo comprimento. O limite de 5 pontos é convenção.

**Filtro de cópia.** Para cada texto, $c$ é o comprimento, em palavras, do maior trecho contíguo que também ocorre na partição de treino. $k$ é o percentil 99 de $c$ entre excertos humanos de teste do mesmo comprimento das amostras. Amostra com $c > k$ é descartada, e a taxa de descarte por ponto, com intervalo de Clopper-Pearson, é relatada.

**Níveis N2 a N4.** Rubrica de 0 a 2 por nível. Um nível conta num ponto quando pelo menos 10 de 12 itens têm mediana dos avaliadores de pelo menos 2. O limite inferior de Clopper-Pearson unilateral de 95% para $x$ sucessos em $n$ itens é o quantil 0,05 da distribuição $\mathrm{Beta}(x, n - x + 1)$, que vale 0,562 para 10 de 12.

**Concordância.** Alfa de Krippendorff com métrica ordinal, só sobre os itens de modelo, com limiar de 0,667, calculado com e sem o autor. Um avaliador sai se não ordenar o excerto humano acima do embaralhado em pelo menos 18 de 20 pares de controle. Ordenar acima significa soma das notas N2, N3 e N4 do excerto humano estritamente maior que a do embaralhado do mesmo par. Empate ou nota faltante contam como falha. O alfa de Krippendorff segue a definição de Krippendorff (2011), com matriz de coincidências e suporte a dados faltantes.

## 11. Sentinela de precisão

As métricas e margens estão na [decisão de precisão](decisao_precisao.md). O erro relativo por tensor é $\|g_{16} - g_{32}\| / \|g_{32}\|$, com $g_{16}$ o gradiente em fp16 desescalado e $g_{32}$ o gradiente em fp32 na mesma máscara e no mesmo lote, avaliado no percentil 95 entre os lotes-sonda de cada faixa de $l/L$.

## 12. Calibração e decisão seletiva

As definições seguem o material de decisores tipados ([pcbrom/laya-jev](https://github.com/pcbrom/laya-jev)) e estão em `slm/calibracao.py`. A unidade é a previsão de um token: no autoregressivo, cada posição dado o prefixo, e na difusão, cada posição mascarada de um sorteio de $l$. A confiança é a maior probabilidade do softmax, e o acerto é o token de maior probabilidade coincidir com o alvo.

**Temperatura.** $T = 1/\beta$, com $\beta$ ajustado por Newton para minimizar a log-verossimilhança negativa média de $\mathrm{softmax}(\beta z)$ na validação. A função é convexa em $\beta$, porque a segunda derivada é a variância média de $z$ sob $\mathrm{softmax}(\beta z)$.

**ECE.** Média, ponderada pela fração de previsões, de $|\bar c_b - \bar a_b|$ em 15 faixas iguais de confiança.

**Diagrama de confiabilidade.** Confiança declarada contra acurácia observada em 10 faixas, com faixas de menos de 20 previsões omitidas. Pontos abaixo da diagonal indicam superconfiança, e acima, subconfiança.

**Decisão seletiva.** Para cada limiar $\tau$, a cobertura é a fração de previsões com confiança de pelo menos $\tau$, e a acurácia seletiva é a acurácia entre elas. O erro do tipo I é aceitar uma previsão errada, e o do tipo II é rejeitar uma certa.

**ROC.** Área sob a curva ROC da confiança como detector de acerto, pela estatística de Mann-Whitney.

**Portão Beta.** Na metade de calibração, o menor limiar cujos itens aceitos atingem o alvo, julgado pelo quantil inferior de 5% de $\mathrm{Beta}(a+1, e+1)$, com $a$ acertos e $e$ erros, e exigindo pelo menos 20 itens aceitos. O limiar é aplicado à metade relatada.

## 13. Curva de deleção

Adaptação, para previsão de próximo token, da curva de deleção usada na análise de decisores tipados, em `slm/delecao.py`.

**Item.** Um trecho do teste com $n = 50$ palavras de contexto, sendo palavra uma sequência máxima sem espaço, e a palavra seguinte. O alvo é o primeiro token da palavra seguinte precedida de espaço.

**Previsão.** No autoregressivo, a entrada é `[BOS]` seguido do contexto, e a previsão é a da posição seguinte. Na difusão, a entrada é o contexto com a posição do alvo e todas as posições seguintes da janela em `[MASK]`. Contextos com mais de $L - 1$ tokens mantêm os últimos $L - 1$. As probabilidades usam a temperatura ajustada na seção 12.

**Decisão e importância.** A decisão $d$ é o token de maior probabilidade com o contexto completo, e $p_0 = p(d)$. A importância da palavra $i$ é $p_0 - p(d \mid \text{contexto sem a palavra } i)$.

**Curva.** Para cada fração $f \in \{0; 0{,}1; \dots; 1\}$, deletam-se as $\mathrm{round}(f n)$ palavras mais importantes e registra-se $\min(p(d)/p_0, 1)$. A curva de referência usa ordens aleatórias, com média de 3 sementes. A AUC é a integral pela regra do trapézio sobre a grade. O valor em $f = 1$, com contexto vazio, mede o viés intrínseco do modelo.

**Relação com a confiança.** Correlações de Spearman e de Pearson entre $p_0$ e a AUC da curva por importância, item a item.

## 14. Gráficos

As figuras são produzidas por `slm/graficos.py` a partir dos arquivos de `runs/` e das análises, em PNG.

| Bloco | Figura | Função | Dados |
|---|---|---|---|
| Treino | Perda, taxa de aprendizado, escala do `GradScaler` e passos pulados | `treino` | `passos.jsonl` |
| Treino | Erro relativo do gradiente e underflow por faixa de $l/L$ | `sentinela` | `sentinela.jsonl` |
| Validação | Bits por byte contra épocas, com intervalo entre sementes | `validacao_epocas` | `avaliacao.json` da grade |
| Validação | Treino contra validação | `treino_validacao` | avaliação na partição de treino e na de validação |
| Teste | $\Delta$ por obra, com o intervalo do bootstrap | `delta_por_obra` | `avaliacao.json` do teste, seção 9 |
| Teste | Bits por byte com e sem nomes | `com_sem_nomes` | `avaliacao.json` do teste |
| Resultados | Novidade de n-gramas, de 1 a 10, contra o texto humano | `novidade_ngramas` | `amostras.jsonl` |
| Resultados | Taxa de descarte por cópia ao longo da grade | `descarte_copia` | `amostras.jsonl` |
| Resultados | Entropia contra perplexidade gerativa por temperatura | `entropia_genppl` | amostras por temperatura |
| Resultados | Níveis de coerência com limite inferior | `niveis` | leitura cega |
| Calibração | Confiança declarada contra acurácia observada, com ECE | `confiabilidade` | seção 12 |
| Calibração | Cobertura contra acurácia seletiva e erros dos tipos I e II | `seletiva` | seção 12 |
| Sensibilidade | Taxa de aprendizado contra regularização, por modo | `sensibilidade_grade` | busca de hiperparâmetros |
| Sensibilidade | $K$, $T_{gen}$, $L$, $V$, precisão e critério de orçamento | `sensibilidade_linha` | corridas dedicadas, por parâmetro |
| Deleção | Curvas por importância e aleatória e dispersão de $p_0$ contra AUC | `delecao` | seção 13 |

As análises de sensibilidade são descritivas e não alteram o critério confirmatório da [decisão de orçamento](decisao_orcamento.md).

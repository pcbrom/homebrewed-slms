# Decisão: precisão numérica no treino, na avaliação e na amostragem

**Estado:** aceita pelo autor em 22/10/2026. **Data:** 22/10/2026. **Veredito do debate adversarial:** condicional, sem objeção fatal remanescente para o escopo, convergido na rodada 3.

## Questão

Como treinar, avaliar e amostrar os dois modos (difusão mascarada e autoregressivo) na GPU T4 do Colab, que não oferece bf16, sem que a precisão numérica distorça a comparação entre eles nem a diferença entre a T4 e a máquina de calibração.

## Decisão

**Treino.** `torch.autocast` em fp16 com pesos mestres em fp32 e `GradScaler` padrão nos dois modos e nas duas máquinas. A perda é calculada em fp32 sobre `logits.float()`, e o backward roda fora do bloco de autocast, como recomenda a documentação do PyTorch. A difusão usa a forma da perda por número de máscaras, $\frac{1}{l}\sum_{i \in S} -\log p_\theta(x^i \mid x_{\setminus S})$ por sequência, com $l \sim U\{1, \dots, L\}$ estratificado no lote e média sobre o lote. A normalização pelo número de tokens mascarados fica proibida.

**Avaliação.** Inteira em fp32, pela forma por número de máscaras, sem limite inferior de $t$.

**Amostragem.** Logits em fp32 fora do autocast e sorteio categórico em float64.

**Atenção.** O kernel do SDPA é fixado em `EFFICIENT_ATTENTION` nas duas máquinas. Se ele não existir em fp16 na T4, as duas máquinas passam para `MATH`. A disponibilidade é conferida antes do portão G1.

**Orçamento por passo aplicado.** Conta apenas o passo em que o otimizador atualizou os pesos. A sequência de lotes é sorteada antes da corrida, e um passo pulado pelo `GradScaler` repete o mesmo lote, o que preserva o pareamento entre braços. FLOPs e tokens ficam iguais entre os modos. O tempo de parede inclui os pulos, e pulos, escala do `GradScaler` e máximo de $|\nabla|$ nos logits ficam registrados por modo.

## Justificativa

O risco de estouro em fp16 está no gradiente que chega aos logits, não na perda, porque o autocast já calcula `cross_entropy` e `log_softmax` em fp32. Num teste em CPU com um token mascarado por sequência, $L = 256$ e escala inicial de $2^{16}$, o máximo de $|\nabla|$ nos logits em fp16 foi o seguinte:

| Forma da perda | Lote 2 | Lote 4 | Lote 64 |
|---|---|---|---|
| Média sobre mascarados, peso $1/t$ | estouro | estouro | estouro |
| Peso $1/t$ dividido por $L$ (MDLM) | estouro | 64.000 | 4.000 |
| Peso $L/l$ dividido por $L$ | 32.768 | 16.384 | 1.024 |

O teto do fp16 é 65.504. A forma por número de máscaras limita o pico a $S/B$, com $S$ a escala e $B$ o lote, e é a NELBO exata sem corte em $t$ (Ou et al., 2024, Teorema 2, e Nie et al., 2025, Eq. 14). A identidade $\int_0^1 \binom{L}{l} t^{l-1}(1-t)^{L-l}\,dt = 1/l$, que liga as duas formas, foi conferida em aritmética racional para todo $l$ de 1 a 256. A única fonte primária sobre treino de difusão mascarada em 16 bits é uma frase do RADD (Ou et al., 2024, apêndice J), que não informa escalonamento nem acelerador. Ela vale como evidência de existência, não como receita.

Rodar fp16 também na 5090, em vez de bf16, isola a precisão como fator único na máquina de calibração. A comparação com fp32 mede o efeito da precisão, e a comparação entre T4 e 5090 mede só o efeito do hardware.

## Sentinela de fidelidade do gradiente

Um lote-sonda fixo por faixa de $l/L$, com as faixas $[1/L, 0{,}01)$, $[0{,}01, 0{,}1)$, $[0{,}1, 0{,}5)$ e $[0{,}5, 1]$, e uma faixa única no autoregressivo. O gradiente em fp16 (desescalado) e em fp32 é medido com a mesma máscara no início, aos 20% do orçamento, a cada 500 passos e no checkpoint final. O aceite é por tensor, em todos os parâmetros, no percentil 95 entre lotes, com as seguintes margens declaradas, sem fonte:

- erro relativo $\|g_{16} - g_{32}\| / \|g_{32}\| \le 0{,}05$, com piso de norma. Tensores com $\|g_{32}\|$ abaixo do piso são relatados à parte e não reprovam.
- erro de cada tensor na difusão até 2 vezes o erro do mesmo tensor no autoregressivo.
- underflow, contado como zero em fp16 e diferente de zero em fp32, em até 1% dos elementos do tensor.
- máximo de $|\text{ativação}|$ por camada até $2^{12}$.
- passos pulados em até 1% após o passo 50, em cada modo, e nenhum NaN.

## Portões, em ordem

| Portão | Onde | Critério | Se falhar |
|---|---|---|---|
| G0 | CPU | Identidade exata e limite do gradiente nos logits | Aprovado |
| G1 | 5090 a 500 W | Sentinela no início e aos 20%, nos dois modos. Custo da sentinela medido | Falha só no autoregressivo indica defeito de código. Falha em $l$ pequeno aciona os remédios em ordem, cada um retestado: `lm_head` e perda em fp32 (custo 2,5 vezes), fp32 puro no modelo 4 × 256, redução do modelo |
| G2 | 5090 | 3 sementes em fp16 e 3 em fp32 por modo, mesmos dados e máscaras. Diferença de bits por byte pareada por semente, por $t$ com 2 graus de liberdade ou por modelo por documento com a semente como cluster, com o intervalo dentro de $\pm m_{prec}$ nos dois modos | Remédios do G1 |
| G3 | T4 contra 5090 | Mesma semente, ordem de dados e máscaras pré-sorteadas. $|\Delta \text{bpb}| \le m_{prec}$, mais os critérios de pulos e NaN. Tempo por passo e MFU registrados | Reduz-se o tamanho do modelo, não a precisão |
| G4 | Corrida final | Sentinela ativa até o checkpoint final | A corrida sai da comparação e volta aos remédios do G1 |

A margem $m_{prec}$ vale um quarto da margem mínima de efeito entre os modos, fixada na [decisão de orçamento](decisao_orcamento.md) antes do G2. Se o custo da sentinela na T4, onde o backward em fp32 é cerca de 8 vezes mais lento, não couber na aula, a frequência dela é reduzida na T4 e mantida na 5090.

## Riscos declarados

- A receita fp16 para difusão mascarada com a forma por número de máscaras não tem receita publicada. A validade dela repousa nos portões G1 a G4 medidos neste projeto.
- As margens da sentinela não têm fonte. São margens declaradas antes da corrida.

## Fora do escopo

- Treino em bf16 na T4, que não o suporta.
- Comparação de precisões como resultado do projeto. A precisão é condição de validade, não objeto de estudo.

## Registro da medida

Em 26/10/2026, na RTX 5090 a 500 W, com a rede de 6 camadas, $d = 384$, $V = 4.096$, $L = 256$, lote 64 e `EFFICIENT_ATTENTION`.

**Portão G1**, sentinela no início, aos 20%, no passo 500 e no fim de 16 épocas, nos dois modos. O pior erro relativo por tensor foi de 0,0025 no autoregressivo e 0,0099 na difusão, contra a margem de 0,05. O underflow máximo foi de 0,7%, contra 1%. A ativação máxima foi de 368, contra $2^{12}$. Não houve passo pulado nem NaN. O critério de erro da difusão até 2 vezes o do autoregressivo no mesmo tensor foi violado em 110 combinações de passo, faixa e tensor, com erros absolutos pequenos, por exemplo 0,0037 contra 0,0008, porque o erro do autoregressivo fica perto do arredondamento do fp16. O critério não tem piso absoluto. O portão G1 está reprovado nesse critério, pendente de decisão do autor.

**Portão G2, rodado como diagnóstico.** Diferença de bits por byte na validação, fp16 menos fp32, pareada por semente, 3 sementes e 16 épocas. Autoregressivo: diferenças de 0,0031, 0,0048 e −0,0063, média 0,0005, IC de 95% por $t$ com 2 graus de liberdade de −0,0143 a 0,0153, desvio entre sementes de 0,0106 em fp16 e 0,0057 em fp32. Difusão: diferenças de −0,0054, −0,0017 e 0,0002, média −0,0023, IC de −0,0094 a 0,0048, desvio entre sementes de 0,0037 e 0,0025. O critério de aceite compara o IC com $\pm m_{prec} = \pm \Delta_{min}/4$, e $\Delta_{min}$ só é congelado no portão G3 do orçamento.

## Emenda de 26/10/2026

Aprovada pelo autor, depois da medida do portão G1. O critério "erro de cada tensor na difusão até 2 vezes o erro do mesmo tensor no autoregressivo" sai da sentinela, porque a razão entre dois erros perto do arredondamento do fp16 não tem piso absoluto e reprova diferenças sem efeito prático. A interação entre precisão e modo passa a ser julgada pela medida de resultado: no portão G2, o IC da diferença de bits por byte entre fp16 e fp32, pareada por semente, deve ficar dentro de $\pm m_{prec}$ em cada modo, com as 3 + 3 corridas já feitas e $m_{prec}$ calculado a partir do $\Delta_{min}$ congelado no portão G3 do orçamento. Os demais critérios da sentinela continuam valendo. Com a emenda, o portão G1 está aprovado. A emenda foi feita depois de ver os dados e fica registrada como tal. Nenhum número novo foi escolhido com base nos dados.

## Julgamento do portão G2

Com $\Delta_{min} = 0{,}0907$ bit por byte congelado no pré-registro `preregistro/m6_delta_min.json`, a margem é $m_{prec} = \Delta_{min}/4 = 0{,}0227$. O intervalo de 95% da diferença de bits por byte entre fp16 e fp32, pareada por semente, vai de −0,0143 a 0,0153 no modo autorregressivo e de −0,0094 a 0,0048 na difusão, os dois dentro de $\pm m_{prec}$. O portão G2 está aprovado nos dois modos, e o treino segue em fp16. Reprodução: `scripts/portao_precisao.py`.

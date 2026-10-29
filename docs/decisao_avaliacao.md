# Decisão: o modelo é bom e útil?

**Estado:** aceita pelo autor em 27/10/2026. **Data:** 27/10/2026. Base: duas pesquisas com fonte, uma sobre métricas semânticas automáticas e outra sobre tarefas, linhas de base e modelos de referência.

## Questão

Um modelo de 12M de parâmetros treinado do zero numa GPU doméstica é bom e útil num domínio especialista? Bits por byte sozinhos não respondem. Falta saber contra o quê o número se mede e para qual tarefa o modelo serve. A literatura não diz se um modelo desse tamanho é útil neste corpus. Ela diz como medir sem enganar, e esta decisão aplica isso.

## Decisão proposta

### 1. Conjunto de avaliação externo no tempo

O corpus B termina em 2022. Leis ordinárias substantivas e alteradoras publicadas a partir de 01/01/2025 formam um conjunto externo, coletado pelo mesmo adaptador e com a mesma limpeza. Nenhum dos modelos comparados pode tê-las visto: o modelo do projeto treina até 2022 e as referências públicas foram publicadas antes de 2025. Esse conjunto é o principal para tarefas e referências. O teste original de 2019 a 2022 (19 leis em 12 grupos, um deles com 53% dos bytes) fica como secundário, porque o Tucano tem contaminação documentada nesse período: o GigaVerbo inclui o LegalPT, com projetos de lei federais, e o índice CC-MAIN-2023-06 tem 191 páginas distintas de leis de 2019 a 2022 do Planalto.

### 2. Referências

- **Piso:** n-grama com suavização de Kneser-Ney modificada, implementado no próprio pacote, com o mesmo propósito didático do BPE escrito à mão e com portão de normalização exata, treinado na mesma partição de treino e sobre o mesmo BPE, com a ordem escolhida na validação entre 3 e 6. É o método clássico para corpus pequeno, e um modelo neural precisa superá-lo para mostrar que aprendeu além de contagens. No BLiMP, o 5-grama fez 61,2 contra 83,0 do GPT-2, de modo que o ganho sobre o n-grama é o número que importa.
- **Régua pública:** TeenyTinyLlama 160M e Tucano 160M, do mesmo tamanho entre si e cerca de 13 vezes maiores que o modelo do projeto, mais o Tucano 630M para mostrar o efeito de escala. Usados só para avaliação, em bits por byte e nas tarefas. Antes de qualquer número, os logits em 16 bits são conferidos contra fp32 numa amostra, e o tokenizador é conferido por ida e volta, por causa do defeito de conversão relatado no Manacá-1B.
- **No corpus A**, a régua entra só em bits por byte, com o risco de contaminação declarado como desconhecido, porque os cartões dos modelos não permitem afirmar nem negar a presença de Machado de Assis.

### 3. Tarefas no corpus B

| Tarefa | Itens | Modo autorregressivo | Modo de difusão | Linhas de base | Métrica |
|---|---|---|---|---|---|
| T1 Lacunas | Número no par "N (extenso)" ou remissão "art. N", trecho de 2 a 8 tokens, em todas as ocorrências de cada artigo completo. Quatro alternativas da mesma categoria que deixam o artigo com o mesmo número de tokens: números do treino no mesmo gênero, remissões a outros artigos citados na mesma lei | Log-verossimilhança exata da sequência inteira | Condicional exata numa passada para um token, estimador condicional do LLaDA com pelo menos 128 amostras e as mesmas máscaras para trechos | n-grama, frequência do candidato, acaso de 25% | Acurácia, e no formato aberto acerto exato e chrF |
| T2 Dispositivo inserido ou próprio | Caput de artigo completo, sem o rótulo "Art. N", com pelo menos 15 palavras. Inserido quando o artigo está inteiro num bloco citado, escrito para dentro de outra norma. Próprio quando é artigo da própria lei, excluídos o objeto do Art. 1º, o artigo que embrulha a alteração, a vigência, a revogação e os vetados | Log-verossimilhança do caput depois do prefixo de inserção ("passa a vigorar com a seguinte redação" e abertura de citação) contra depois do prefixo de artigo próprio, com calibração por entrada vazia | O mesmo contraste pela condicional da difusão com o caput mascarado | Classe majoritária, comprimento, TF-IDF com regressão logística, Bayes com um n-grama por classe | Acurácia balanceada, por validação cruzada agrupada no corpus inteiro e no conjunto externo |
| T3 Texto adulterado | Artigo completo sem o rótulo, com no máximo 254 tokens, no original e numa versão com um operador de uma de quatro famílias: termo trocado numa lista fechada de pares com a mesma gramática, número trocado no par "N (extenso)" com o extenso refeito no mesmo gênero, remissão trocada por outro artigo citado na mesma lei, e conteúdo de dois dispositivos do mesmo tipo trocado com os rótulos no lugar | Prefere o original se a log-verossimilhança for maior | Posições diferentes mascaradas, NELBO com máscaras comuns no embaralhamento | n-grama, TF-IDF com regressão logística lendo um texto por vez, acaso de 50% | Acurácia |
| T4 Continuação | Caput de artigo completo cortado na metade das palavras, com o prefixo como condição e a continuação real como alvo | Bits por byte da continuação real dado o prefixo | O mesmo, como limite superior declarado | Geração do n-grama, cópia da continuação do dispositivo de treino de maior prefixo comum | Bits por byte, e chrF, ROUGE-L e juiz semântico sobre geração no comprimento real |

Itens com sobreposição de 13 tokens com o treino saem antes da medida.

T2 foi redefinida em 28/10/2026, com aprovação do autor e antes de qualquer item construído, porque a classe da lei inteira se reduz a comprimento (`scripts/sonda_t2.py`, saída em `resultados/sonda_t2.json`). Depois das remoções previstas, o corpo mediano de uma alteradora tem cerca de 700 bytes e o de uma substantiva de 3,2 a 4,4 KB, e o corte só por comprimento acerta de 0,80 a 0,81 em acurácia balanceada. O objeto do Art. 1º, exigido pela LC 95/1998, art. 7º, repete a alteração em 27% a 34% das alteradoras. No nível do caput, o comprimento acerta 0,52 no corpus B e 0,53 no conjunto externo, as palavras-pista de remissão ficam equilibradas entre as classes, e o TF-IDF com regressão logística em validação cruzada agrupada no treino acerta 0,574, com sinal temático. O artigo que embrulha a alteração fica de fora porque a forma dele entrega o rótulo. Os blocos citados são lidos por contagem de profundidade das aspas, para que uma citação aninhada não feche o bloco antes da hora, e a lei cujas aspas não fecham sai inteira (52 no corpus B e 8 no conjunto externo, em geral memoriais descritivos que usam a aspa de fechamento como sinal de segundos). Caput acima de 254 tokens também sai (13 e 10). Restam 1.945 itens no corpus B, 510 deles inseridos, e 1.494 no conjunto externo, 369 inseridos. Depois do filtro de 13 tokens, o conjunto externo fica com 760 itens, 155 inseridos em 42 grupos, e a validação e o teste do corpus B ficam com 3 e 15 inseridos. Por isso T2 se mede no conjunto externo. O rótulo sai da estrutura do texto e deixa de depender da auditoria das 72 ementas.

T3 foi detalhada em 28/10/2026, com aprovação do autor e antes de qualquer medida (`slm/tarefas.py`, `scripts/sonda_t3.py`, saída em `resultados/sonda_t3.json`). Cada artigo recebe no máximo um operador, o primeiro aplicável em ordem de sha256. Os pares de termos são poderá e deverá, poderão e deverão, no mínimo e no máximo, Estados e Municípios, mensal e anual, e a troca não se aplica quando o termo de destino já aparece no artigo, para não repetir um ente numa enumeração. Nas famílias de termo e de número, as trocas preservam a frequência de cada valor entre os textos adulterados: cada par de termos tem o mesmo número de trocas nos dois sentidos, e os números são permutados ciclicamente dentro de cada classe de gênero. Sem isso, um decisor que lê um texto por vez aprenderia que "deverá" indica adulteração, porque "poderá" é quase duas vezes mais frequente no corpus. O TF-IDF com regressão logística lendo um texto por vez acerta 0,492 em validação cruzada agrupada no treino, entre 0,38 e 0,55 por família, o que indica ausência de pista de superfície. São 646 itens no corpus B, 595 deles no treino, e 492 no conjunto externo, em 115 grupos. O filtro de 13 tokens vale para o artigo inteiro, nas duas versões, e deixa no conjunto externo 167 itens em 66 grupos (70 de termo, 49 de remissão, 34 de embaralhamento e 14 de número). Um filtro restrito ao trecho em volta da edição deixaria 374 itens em 100 grupos, e foi preterido em favor da proteção mais conservadora contra memória.

T1 ficou só com número e remissão em 28/10/2026, com aprovação do autor. O BPE do corpus B quebra quase todo termo jurídico em vários tokens, e nenhuma categoria fechada tem quatro opções de um token só ("dias" e "anos" têm um token, "meses" dois, "União" um e "Municípios" cinco), então a família de termo sai e fica declarada como limitação do tokenizador. Com o filtro do artigo inteiro, T1 fica com 89 lacunas em 23 grupos no conjunto externo e é relatada com intervalo, sem teste confirmatório, porque esse número de grupos não sustenta o teste do sinal.

Itens construídos por `scripts/construir_tarefas.py` em `tarefas/B/`, com contagens em `resultados/tarefas_B.json`. A construção é determinística, e os itens de treino ficam inteiros para o ajuste do decisor. No conjunto externo, depois do filtro de 13 tokens:

| Tarefa | Itens | Grupos | Composição |
|---|---|---|---|
| T1 Lacunas | 89 | 23 | 63 remissões, 26 números |
| T2 Dispositivo inserido ou próprio | 760 | 148 | 155 inseridos, 605 próprios |
| T3 Texto adulterado | 167 | 66 | 70 termos, 49 remissões, 34 embaralhamentos, 14 números |
| T4 Continuação | 508 | 133 | |

Com esses números, o teste do sinal só teria 80% de poder em T3 se 72% dos grupos favorecessem o modelo, no pior caso de Holm, e em T1 se 86% favorecessem. A emenda de 28/10/2026, aprovada pelo autor e congelada em `preregistro/m14_tarefas_B_emenda.json` antes de qualquer medida nos itens novos, amplia o conjunto de medida sem afrouxar o filtro. O conjunto passa a ser `B_externo_atos`: leis ordinárias, leis complementares e decretos de 2025 e 2026, das classes substantiva e alteradora, num só agrupamento, de modo que um decreto que regulamenta uma lei do conjunto cai no grupo dela. São 1.094 atos (300 leis, 27 leis complementares e 767 decretos) em 855 grupos, com o maior grupo em 7,8% dos bytes, e as 300 leis têm texto idêntico ao do conjunto anterior. T3 passa a gerar um item por família aplicável em cada artigo, e o TF-IDF lendo um texto por vez segue em 0,507 no treino. Os gêneros são relatados à parte como descrição. Antes da emenda, só os dois pisos tinham sido medidos, no conjunto de leis, e os valores estão declarados na própria emenda. O conjunto `B_externo` de leis continua sendo o da curva de dados, pré-registrada nele.

| Tarefa | Itens | Grupos | Proporção de grupos a favor para 80% de poder (Holm, pior caso) |
|---|---|---|---|
| T1 Lacunas | 337 | 130 | 0,655 |
| T2 Dispositivo inserido ou próprio | 3.095 (313 inseridos) | 643 | 0,570 |
| T3 Texto adulterado | 1.185 | 469 | 0,581 |
| T4 Continuação | 2.251 | 615 | 0,571 |

A segunda emenda, do mesmo dia (`preregistro/m14_tarefas_B_emenda2.json`), corrige o filtro de janela de T2, que media o caput sozinho, quando a pontuação gerativa põe antes dele o prefixo de inserção, de 20 tokens. O item de T2 passa a entrar só quando o caput com esse prefixo cabe em 254 tokens, o que tira 13 dos 4.960 itens (4 do conjunto externo, 8 do treino e 1 da validação). A medida interrompida por esse limite tinha gravado escores dos pisos e do modo autorregressivo nos itens da primeira emenda, que não foram lidos e foram recalculados.

### 4. Qualidade semântica das amostras

- **Principal:** juiz local, Qwen3.8-27B pelo llama-swap, com a rubrica N2 a N4 da leitura cega e escore ponderado pelas probabilidades no estilo G-Eval, aplicado só depois do filtro de cópia. É a única família com correlação publicada no nível do item, que é o nível da leitura humana do projeto. O desenho segue o painel de juízes com rubrica ancorada do TraCE-LLM.
- **Portão contra a leitura cega**, com limiares congelados antes de medir: limite inferior do IC de 95% do Spearman por item acima de 0,3; pelo menos 80% do teto humano, medido deixando um avaliador de fora; AUC de pelo menos 0,95 separando texto humano de embaralhado e de repetitivo. O juiz só entra no artigo como métrica se passar. O portão depende da leitura cega do M9.
- **Diagnósticos, sem decisão:** MAUVE por condição, porque a validação publicada usa pelo menos 1.000 amostras e textos de 256 tokens ou mais, e perplexidade gerativa sempre ao lado da entropia, porque texto degenerado tem perplexidade baixa.

### 5. Estatística

A unidade é o grupo de leis, não o item. Todos os itens que passam pelo filtro entram, e a média macro por grupo é a estatística principal, o que dá a cada grupo o mesmo peso. Intervalos por bootstrap selvagem por grupo. Comparações pareadas por teste do sinal sobre os grupos. Análise refeita sem o maior grupo. A correlação intragrupo de cada tarefa é relatada como descrição. O plano aprovado previa uma cota igual de itens por grupo, tirada da correlação intragrupo medida num piloto, para que a amostragem proporcional aos bytes não reduzisse o teste original a cerca de 3,3 grupos efetivos. A cota saiu em 28/10/2026, com aprovação do autor e antes de qualquer medida, porque a média macro por grupo já iguala o peso dos grupos, porque os itens construídos são baratos de pontuar, e porque o piloto exigiria itens fora do conjunto externo, que a validação e o teste do corpus B não têm depois do filtro (T2 com 76 itens em 16 grupos, T3 com 13). O critério de leitura de cada comparação é congelado com hash antes da medida. O modelo do projeto "supera o piso" numa tarefa quando o teste do sinal sobre os grupos rejeita a 0,05 a favor dele. A posição frente à régua pública é relatada com intervalo, sem aprovação nem reprovação.

### 6. Curva de dados

Aprovada pelo autor em 27/10/2026, depois de o piso n-grama do corpus A ficar a menos de $\Delta_{min}$ do melhor Transformer. A pergunta é a partir de quanto texto o Transformer passa a valer a pena. O modo autorregressivo, com os hiperparâmetros do corpus escolhidos no M6, e o n-grama treinam em frações aninhadas dos grupos de treino do corpus B (1/8, 1/4, 1/2 e todo o treino). Em cada fração, o Transformer roda com 16 e 64 épocas e 3 sementes, e fica o orçamento de menor média na validação. A ordem do n-grama é escolhida na validação entre 3 e 6. Os dois e a mistura deles, com o peso ajustado na validação por semente, são medidos no conjunto externo. O Transformer vale a pena numa fração quando o limite inferior do IC de 95% da vantagem sobre o n-grama passa $\Delta_{min}$ do corpus B. O desenho é congelado com hash antes das corridas (`scripts/curva_dados.py preregistrar`), o que exige o $\Delta_{min}$ do M6 do corpus B. O custo estimado é de cerca de 20 minutos de GPU.

### 7. Do gerador ao decisor tipado

Aprovado pelo autor em 27/10/2026, no mesmo artigo, porque usa o mesmo corpus, os mesmos checkpoints e as mesmas tarefas. A pergunta é se o modelo treinado em casa serve de base para um decisor tipado no estilo do Laya, que recebe o texto e devolve a distribuição de probabilidade sobre opções fixas numa passada, sem gerar texto, e qual dos dois modos rende o melhor decisor. A difusão mascarada é bidirecional e por isso candidata natural a encoder. O modo autorregressivo entra como controle.

- **Encoder:** os checkpoints do M7 do corpus B nos dois modos, com média dos estados da última camada e uma cabeça linear sobre as opções, ajustados com os rótulos do treino. O modo de difusão lê o texto sem máscara.
- **Tarefas:** T2 (dispositivo inserido ou próprio) e T3 (texto adulterado, cada texto julgado sozinho como original ou adulterado), com os mesmos itens, rótulos e filtros da seção 3. Os itens de ajuste de T3 são adulterações geradas sobre as leis de treino, com os mesmos operadores.
- **Entrada:** todos os decisores leem o mesmo item. Em T2 todo caput cabe na janela de 256 tokens do modelo do projeto, porque os maiores saem na construção. Em T3, o item que passar da janela é cortado no mesmo ponto para todos, e o Laya com a janela inteira dele entra como leitura secundária, para medir quanto o corte custa.
- **Piso:** TF-IDF com regressão logística, o equivalente do piso n-grama numa decisão, já previsto como linha de base de T2.
- **Régua pública:** o Laya multilíngue (mmBERT-base, 322M, Apache 2.0) ajustado nos mesmos rótulos e com o mesmo orçamento de ajuste, e o Laya sem ajuste como referência. Ao contrário das réguas gerativas, ele precisa de ajuste, porque a cabeça é por tarefa.
- **Métricas:** acurácia balanceada, ECE antes e depois de temperatura ajustada por validação cruzada agrupada no treino, já que a validação do corpus B fica com poucos itens de T2 depois do filtro, e cobertura da resposta seletiva com 95% de acerto. Avaliação no conjunto externo de 2025 em diante, com a estatística por grupo da seção 5. O modelo do projeto supera o piso quando o teste do sinal sobre os grupos rejeita a 0,05 a favor dele. A posição frente ao Laya é relatada com intervalo, sem aprovação nem reprovação.
- **Fora:** o extrator no estilo do Needle (geração sob gramática do esquema com verificação de âncora), por falta de dados de extração rotulados em português e de régua pública confiável sem ajuste. Fica declarado nas limitações.

### 8. Ordem e custo

1. Coleta das leis de 2025 em diante e auditoria da regra de classe nelas (rede e CPU).
2. Implementação e portões do n-grama (CPU).
3. Construção dos itens e pré-registro.
4. Medida do modelo do projeto e das réguas (GPU, minutos), depois do M7 do corpus B.
5. Decisor tipado nos dois modos, no piso e no Laya (GPU, minutos), depois dos itens de T2 e T3 e do M7 do corpus B.
6. Juiz semântico, depois da leitura cega do M9.

### 9. Resultado no conjunto externo e desvio declarado

Medido em 28/10/2026 sobre os itens da segunda emenda, com o modelo do projeto no melhor ponto de validação de cada modo (autorregressivo em 64 épocas com 5 sementes, difusão em 256 épocas com 6 sementes) e números em `resultados/m14_B.json`. O modelo tem 11,1 milhões de parâmetros, e cada corrida de treino no melhor ponto leva de 90 a 440 segundos numa RTX 5090 a 500 W.

| Pontuador | T1 acerto | T2 acurácia balanceada | T3 acerto | T4 bits por byte |
|---|---|---|---|---|
| n-grama (piso) | 0,241 | 0,502 | 0,497 | 1,355 |
| Autorregressivo | 0,399 | 0,467 | 0,559 | 1,425 |
| Difusão | 0,310 | 0,502 | 0,550 | 1,852 |
| TeenyTinyLlama 160M | 0,269 | 0,541 | 0,567 | 0,791 |
| Tucano 160M | 0,294 | 0,510 | 0,458 | 0,737 |
| Tucano 630M | 0,322 | 0,498 | 0,538 | 0,647 |

Na trilha gerativa, o modo autorregressivo supera o piso em T3 (diferença de 0,062, IC de 95% [0,014, 0,110], 202 de 356 grupos a favor, p de Holm 0,025). A difusão tem intervalo acima de zero em T3 (0,053, [0,015, 0,091]), mas o teste do sinal não rejeita (p de Holm 0,13). Em T4 os dois modos ficam abaixo do piso, o autorregressivo por 0,070 bits por byte e a difusão por 0,497, e a leitura restrita às leis repete a ordem (1,523 contra 1,435 do piso). A análise sem o maior grupo não muda nenhuma leitura.

Na trilha do decisor, nenhum decisor supera o TF-IDF (0,544 em T2 e 0,573 em T3). Os decisores sobre os dois modos ficam em 0,499 e 0,488 em T2 e em 0,501 e 0,514 em T3, e o Laya ajustado fica em 0,485 e 0,500. Em T3 o ajuste não separa as classes nem na validação cruzada do treino (acurácia de 0,49 a 0,51 em todas as taxas, temperatura no limite superior da busca). A resposta seletiva, com limiar fixado fora da amostra no treino para 95% de acerto, cobre em média 0,90% (autorregressivo) e 0,05% (difusão) dos itens externos de T2, com acerto de 0,89 e 0,67 nos itens cobertos, e 2,29% e 0,59% em T3, com acerto de 0,51 e 0,54. O limiar não se transfere para o conjunto externo, e no Laya ajustado ele não cobre nenhum item.

**Desvio declarado depois da medida, em T2.** O critério pré-registrado lê o teste do sinal sobre a contribuição de cada grupo para a acurácia balanceada. No conjunto externo, 547 dos 643 grupos de T2 só têm itens da classe própria, e nesses grupos a diferença entre dois pontuadores mede apenas o acerto na classe majoritária. O teste rejeita então com cerca de 85% dos grupos a favor para qualquer pontuador, com diferença agregada positiva ou negativa (difusão com +0,0004, autorregressivo com −0,034). O teste não mede a métrica da tarefa, e T2 passa a ser lido pelo intervalo da diferença na acurácia balanceada, sem declaração de superioridade: nenhum modo supera o piso em T2, e o autorregressivo fica abaixo dele (IC [−0,064, −0,005]). A saída do critério original permanece em `resultados/m14_B.json`. Nas demais tarefas o teste do sinal e o intervalo apontam no mesmo sentido.

**Outras declarações.**

- A grade de C do TF-IDF, de 0,01 a 100, foi fixada na primeira emenda, depois do pré-registro original e antes de qualquer medida no conjunto ampliado.
- O decisor usa AdamW com os valores padrão do PyTorch, taxa constante e entropia cruzada ponderada pelo inverso da frequência das classes.
- O tokenizador das réguas gerativas descarta º, ª e a ligadura ﬁ, o que corresponde a 0,40% dos bytes das continuações de T4. Cobrar 16 bits por caractere descartado moveria as réguas em no máximo 0,03 bit por byte, contra uma distância de 0,56 a 0,71 até o piso.
- A leitura secundária do Laya com a janela inteira não se aplica, porque o filtro de construção deixa todo item dentro da janela de 256 tokens.
- A taxa do modo autorregressivo no corpus B veio da quina da grade do M6 (3e-4).

## Riscos declarados

- O conjunto externo tem menos leis do que um ano completo do corpus, e o número de leis de 2025 em diante só é conhecido depois da coleta.
- A deriva de tema entre 2019 a 2022 e 2025 em diante pode baixar o desempenho de todos os modelos no conjunto externo, o que se mede comparando com o teste original.
- A distância entre a NELBO e a verossimilhança verdadeira pode diferir entre candidatos e alterar a escolha da difusão nas tarefas. A literatura não traz número para este tamanho de modelo.
- A régua pública em 16 bits pode estourar ativações. A conferência contra fp32 vem antes.
- O corte na janela de 256 tokens pode tirar do decisor a parte do texto que decide em T3, o que a leitura secundária do Laya com a janela inteira mede.
- T2 no nível do caput é difícil para o piso (0,574 no treino), e com todos os métodos perto do acaso a tarefa separa pouco. A posição frente ao Laya mostra se há sinal que um encoder maior capture.
- Em T2 a classe inserida aparece em só 42 grupos do conjunto externo depois do filtro, e T3 fica com 167 itens em 66 grupos, o que limita o teste do sinal sobre os grupos. O piloto de correlação intragrupo diz se o poder basta.
- Os vieses documentados do juiz LLM (nota mais alta fora do inglês, viés positivo, kappa de 0,32 a 0,53 em texto jurídico em português) são a razão do portão contra a leitura humana.

## Fora do escopo

- Conjuntos jurídicos externos rotulados para tarefas de outra natureza (reconhecimento de entidades, decisões de tribunais).
- Ajuste fino das réguas públicas gerativas. O Laya é ajustado porque a cabeça de decisão é por tarefa.

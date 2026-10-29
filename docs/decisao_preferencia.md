# Decisão: fonte de preferência e desenho do pós-treino (etapa 3)

**Estado:** aceita pelo autor em 26/10/2026. **Data:** 26/10/2026. **Veredito do debate adversarial:** condicional, sem objeção fatal remanescente, convergido na rodada 2, com três portões exigidos pelo crítico e incorporados abaixo (P-dose, P-piso e concordância na ordenação).

## Questão

Qual fonte de preferência constrói os pares do pós-treino por DPO nos dois modos de geração, e como a etapa mede coerência e calibração antes e depois.

## Decisão

**Fonte principal: preferência humana.** É o ideal metodológico fixado pelo autor. Os avaliadores ordenam os 4 candidatos de cada prefixo, o que dá 6 pares por prefixo, como na seção C.1 de Rafailov et al. (2023). São 125 prefixos de treino por modo, com 20% de sobreposição entre leitores para medir a concordância. A unidade independente é o prefixo, porque os 6 pares de uma ordenação são transitivos.

**Recompensa programática como braço comparativo, não como fonte.** Sobre os mesmos candidatos, uma recompensa derivada só da partição de treino ordena os pares: a fração de trigramas de palavras atestados no treino, com teto na média humana de teste, mais a taxa de léxico, descartando candidatos com trecho copiado maior que o limiar $k$ do M9. Ela não mede N3 nem N4. A concordância entre a recompensa e o juízo humano dentro do mesmo prefixo é relatada.

**Controle com rótulos embaralhados.** As ordenações humanas com rótulos permutados, no mesmo volume, separam o efeito da preferência do efeito de qualquer atualização dos pesos.

**Referência.** Em cada modo, o modelo de menor bits por byte de validação da busca do M6, que deixa os dois modos no mesmo regime. Com as corridas da busca, isso significa o autorregressivo de 16 épocas e a difusão de 256 épocas. O autorregressivo de 256 épocas está sobreajustado (ECE de 0,315 com $T = 1$, contra 0,029 a 0,033 no de 16 épocas, e 0,016 a 0,018 na difusão de 256 épocas, em 20 mil previsões de validação) e confundiria o efeito do DPO com o do sobreajuste.

**DPO.** No modo autorregressivo, com a log-probabilidade exata. Na difusão, com a estimativa da NELBO e as três técnicas de redução de variância do VRPO (Zhu et al., 2025, seção 3.2), com $n = 8$, alocação ótima e amostragem antitética. O treino otimiza o substituto baseado na NELBO, e o limite de viés do Teorema 1 de Zhu et al. vale em relação a esse substituto, não ao DPO com a verossimilhança verdadeira. $\beta = 0{,}1$, o valor padrão do DPO, pré-registrado, com 0,05 e 0,5 apenas descritivos. Uma época, como no LLaDA 1.5. Os pares saem dos candidatos da semente 0 e são aplicados às 10 sementes, com precedente na seção 6.1 de Rafailov et al.

## Medida antes e depois

**Calibração.** ECE com $T = 1$, temperatura ajustada, ECE com a temperatura, diagrama de confiabilidade, ROC, curva de deleção, bits por byte e KL até a referência, no teste, pareados por semente. Equivalência de ECE por TOST, com margem de 2 desvios do ECE de base entre as 10 sementes, congelada antes do DPO. A diferença de ECE é estimada nos dois sentidos, sem hipótese direcional, porque as fontes não sustentam direção para DPO em calibração de próximo token: a Figura 8 do relatório do GPT-4 mede calibração de resposta no MMLU num modelo treinado por PPO, e Tian et al. (2023) apenas relatam que estudos sugerem a perda.

**Coerência.** Leitura cega com a rubrica congelada, em 48 prefixos de teste por modo, da referência, do braço humano e do controle embaralhado. Desfecho primário: diferença pareada por prefixo da soma de N2 a N4, com efeito mínimo detectável $d = 0{,}46$ a poder 0,8, com correção de Holm a 0,025. Nenhum tamanho de efeito é presumido. A recompensa programática é medida só automaticamente.

## Portões, em ordem

| Portão | Critério | Se falhar |
|---|---|---|
| P0 | Pré-registro com hash: braços, $\beta$, dose, margem do TOST, desfecho primário e critério de concordância na ordenação (τ de Kendall entre leitores nos prefixos sobrepostos). Depende do M6, do M7 e do M9 | Nada roda |
| P-dose | Antes de qualquer coleta humana, o DPO com os pares programáticos na dose final separa-se do controle embaralhado, com intervalo excluindo zero, em KL até a referência e em acurácia de preferência retida. A dose inicial é 125 prefixos, uma época, com taxa de aprendizado constante igual à mínima da referência, cerca de 12 passos contra 694 do pré-treino | A dose sobe, em épocas e depois em taxa de aprendizado, por regra declarada no pré-registro |
| P-piso | No piloto de 80 itens ou no lote de 16 épocas do M9, a soma N2 a N4 da referência fica fora do piso e os leitores concordam na ordenação. A referência autorregressiva tem taxa de léxico de 0,880 contra 0,962 no texto humano, 8,2 pontos abaixo, além do limite de 5 pontos do N1 | N2 passa a ser o desfecho primário, declarado antes da leitura |
| P-cap | O autor decide com os segundos por item medidos no M9 e no piloto e com a disponibilidade declarada dos leitores | Recuo descrito abaixo |
| P1 | Na difusão, menos de 25% dos candidatos acima da média humana de trigramas atestados, para que a recompensa programática não sature | A recompensa é revista antes de comparar |
| P2 | Concordância entre a recompensa programática e os humanos dentro do prefixo, por τ de Kendall com teste unilateral. Poder de 0,86 com 50 prefixos se a concordância verdadeira for 0,60, em simulação com latente normal, sem dado real | Decide só a interpretação do braço programático |
| P3 | Alfa de Krippendorff de pelo menos 0,667 e filtro de 18 em 20 pares de controle | A leitura é refeita com os leitores aprovados |
| P4 | Durante o treino, a 25%, 50% e 100%, o desvio da estimativa da diferença de log-razões fica abaixo da metade do seu valor médio absoluto, em 128 pares retidos com 32 repetições | $n$ passa a 16 e depois a 32 |
| P5 | KL do braço humano acima do controle embaralhado, com acurácia retida relatada | O efeito de coerência não é atribuído à preferência |
| P6 | Teto de 4 h de GPU, com estimativa abaixo de 1,5 h | Corte de sementes, declarado |
| P7 | Versão reduzida reproduzida na T4 | A aula usa checkpoints pré-treinados |

## Carga humana

Cerca de 768 itens por leitor na etapa 3, 2,4 vezes o M9, ou cerca de 1.088 somados ao M9. Os cortes, em ordem, são tirar o controle embaralhado da leitura (672 itens) e depois reduzir a 32 prefixos de teste (efeito mínimo detectável de 0,57). Sem o controle na leitura, a afirmação sobre coerência fica restrita a DPO contra a referência. Com 20% de sobreposição, 80% das ordenações vêm de um único leitor, e o ruído do rótulo é medido em cerca de 25 prefixos por modo.

## Recuo e declaração de limitação

Se a capacidade medida não cobrir o braço humano, a recompensa programática passa a braço principal, sobre 2.000 prefixos, e a validação humana usa os prefixos que couberem. Abaixo de 50 prefixos de validação por modo, o braço programático é relatado como otimização de um substituto, sem afirmação sobre coerência. A metodologia então declara: os pares vêm de recompensa programática por limitação de recursos humanos, medida em segundos por item, e não por equivalência com a preferência humana, que é o ideal. A recompensa substitui N1 e N2 e não mede N3 nem N4. É determinística e viola a premissa estocástica do modelo de Bradley-Terry. A versão humana mudaria o alvo, para coerência de parágrafo, o ruído dos rótulos, medido pela concordância, e o custo.

## Dados e consentimento

As ordenações humanas entram no repositório com os avaliadores codificados, para que o notebook rode o DPO sem leitores ao vivo. O consentimento escrito dos avaliadores para isso é obtido antes da coleta.

## Pendências fora deste documento

- Azar et al. (IPO) só será citado depois de registrado e conferido em `referencias/`.
- A taxa de aprendizado do DPO não tem fonte nesta escala e é fixada pelo portão P-dose.

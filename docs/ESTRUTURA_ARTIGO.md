# Estrutura do artigo tutorial (M12)

**Estado:** aprovada pelo autor em 27/10/2026. O manuscrito é escrito em inglês, para o arXiv. A comparação entre bytes e BPE com computação igual usa só o tokenizador trivial de bytes do contrato, sem nada da proposta de tokenizador.

## Tese do artigo

Enquadramento definido pelo autor: escrutínio na construção artesanal de um SLM para domínio especialista. Um modelo de linguagem pequeno para um domínio estreito pode ser construído à mão, peça por peça, do corpus ao tokenizador e aos dois modos de geração, e cada peça só merece confiança depois de passar por uma medida que poderia reprová-la. O artigo mostra esse escrutínio em ação: os portões, os pré-registros e os defeitos que só a medida revelou, inclusive nos próprios instrumentos de medida. Os resultados sobre os dois modos, em especial o comportamento quando o mesmo texto é repetido muitas vezes, aparecem como o que o escrutínio permite afirmar, e não como a tese.

## Seções

1. **Introdução.** A lacuna: tutoriais mostram o caminho que funcionou e escondem os erros. A pergunta de origem: quanto treino até a saída ficar coerente. As contribuições: o pacote e os notebooks, a formulação exata da difusão mascarada na forma por número de máscaras, o protocolo com portões e pré-registro, o catálogo de armadilhas medidas e as observações sob repetição de dados.
2. **Fundamentos.** Verossimilhança autorregressiva, difusão mascarada e a NELBO (Ou et al., MDLM, RADD, LLaDA), bits por byte como régua independente do tokenizador e a leitura assimétrica imposta pelo limite superior.
3. **O pipeline, caixa por caixa.** As 11 caixas, do adaptador de domínio à leitura, com os blocos de construção matemáticos condensados dos notebooks.
4. **Protocolo de medida.** Portões, pré-registro com hash, busca de hiperparâmetros, $\Delta_{min}$, piloto e número de sementes, bootstrap em dois níveis, cobertura e o teste de aleatorização como recurso. As três emendas são listadas com data e motivo.
5. **Armadilhas medidas.** O núcleo didático: vazamento de comprimento, variância das duas formas da perda, estouro em fp16 e a sentinela, a sentinela que media dropout em vez de precisão, ruído de Gumbel truncado em fp16, logit da máscara, desempate do BPE, regra de vocabulário, vazamento de origem na leitura cega e cópia do treino.
6. **Resultados.** A grade descritiva e o ponto confirmatório, a calibração e a curva de deleção por corrida, os níveis de coerência da leitura cega (ou a lacuna declarada, se a leitura não estiver pronta) a reprodução na T4 e o decisor tipado construído sobre os checkpoints dos dois modos, com acurácia, calibração e resposta seletiva contra o TF-IDF e o Laya.
7. **Observações sob repetição de dados.** Rotuladas como observação, sem afirmação geral: o autorregressivo aprende mais rápido e sobreajusta mais cedo, a difusão tolera muito mais repetições, o melhor ponto de cada modo fica próximo, e o efeito da repetição aparece também na comparação entre bytes e BPE com computação igual. A relação com Prabhudesai et al. (2025) é discutida.
8. **Limitações.** Escala de 12 milhões de parâmetros, corpus principal único, a NELBO como limite superior, a cobertura do bootstrap abaixo do mínimo, as emendas, a dependência da leitura humana, o extrator no estilo do Needle fora do escopo por falta de dados rotulados em português e o que não se transfere para escala.
9. **Reprodutibilidade.** Repositório, DOI do Zenodo, notebooks no Colab, cookbook e comandos.

Apêndices: deduções completas, tabela de portões, busca de hiperparâmetros e detalhes dos testes de enumeração.

## Figuras previstas

As figuras da seção 14 da SPEC: curvas de treino, sentinela, bits por byte por época nos dois modos, treino contra validação, $\Delta$ por obra, efeito dos nomes próprios, novidade de n-gramas, descarte por cópia, níveis de leitura, confiabilidade, decisão seletiva, sensibilidade e curva de deleção.

## Fora do artigo

- A proposta de tokenizador do autor, publicada à parte em [decisao_tokenizador_codigo.md](decisao_tokenizador_codigo.md). O artigo usa só o controle por bytes do mesmo experimento.
- A etapa 3 (pós-treino por preferência), que pode virar um segundo artigo ou uma versão posterior.

## Pendências antes do manuscrito

0. Corridas treinadas no corpus B (legislação federal), que é o domínio especialista do projeto: pelo menos a grade descritiva e o ponto confirmatório, com a mesma bancada. Decisão pendente do autor: reaproveitar os hiperparâmetros do corpus A ou fazer uma busca própria no B.
1. Ponto confirmatório refeito com a sentinela corrigida e portão G4 julgado (concluído em 27/10/2026).
2. Calibração, curva de deleção e figuras por corrida do M7.
3. Leitura cega (M9), dependente dos dois avaliadores. Sem ela, a primeira versão declara a lacuna.
4. Reprodução na T4 pelo autor (M8).

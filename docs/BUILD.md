# Plano de construção

O projeto avança em marcos, sem datas, em sessões curtas e diárias, nesta ordem: etapa 1 (M1 a M9), etapa 2 (M10), etapa 3 de pós-treino por preferência (M11), artigo tutorial para o arXiv (M12) e, por último, abertura do repositório e DOI no Zenodo (M13). O repositório só se torna público depois do artigo. Cada marco termina num portão verificável, e nenhum começa antes que os portões de que depende estejam aprovados.

## Marcos

| Marco | Objetivo | Entregáveis | Governado por | Encerra com |
|---|---|---|---|---|
| M1 Infraestrutura | Ambiente reprodutível e esqueleto do pacote | Ambiente com versões fixadas, `slm/` com um módulo por caixa, `tests/`, integração contínua local por `pytest` | [ARQUITETURA](ARQUITETURA.md), [CONTRATO_DADO](CONTRATO_DADO.md) | Testes vazios rodando, vazão da GPU visível ao PyTorch |
| M2 Corpus | Corpora A e B limpos, divididos e versionados | `corpus/A` e `corpus/B` com manifesto, documentos, divisão, nomes e aviso | [decisão de corpus](decisao_corpus.md) | Corpus G0 a G3 |
| M3 Tokenizador | Tokenizador de referência treinado e contrato testado | `tokenizador/<corpus>/`, BPE didático, testes U1 a U6 e específicos de BPE | [decisão do tokenizador](decisao_tokenizador.md) | Tokenizador G0 a G2 e G5 |
| M4 Modelo em CPU | Janelamento, mascaramento, rede, perdas, geração e avaliação corretos em escala de brinquedo | Caixas 3 a 8 implementadas, testes de enumeração | [decisão de formulação](decisao_formulacao.md), [SPEC](SPEC.md) | Formulação G0 e G1, bloqueantes |
| M5 Calibração | Precisão validada e vazão medida na 5090 | Sentinela, corrida fp16 contra fp32, vazão dos dois modos | [decisão de precisão](decisao_precisao.md) | Precisão G1 e G2, orçamento G1 e G2 |
| M6 Hiperparâmetros e piloto | Configuração escolhida, $\Delta_{min}$ congelado e número de sementes calculado | Busca nos pontos de 16 e 256 épocas, pré-registro, piloto de 10 sementes | [decisão de orçamento](decisao_orcamento.md) | Orçamento G3 e G4 |
| M7 Comparação e análises | Grade descritiva, ponto confirmatório, calibração, curva de deleção, sensibilidade e gráficos | Corridas da grade, avaliação, intervalo de $\Delta$, `calibracao.json` e `delecao.json` por corrida, figuras da seção 14 da SPEC | [decisão de orçamento](decisao_orcamento.md) | Orçamento G5 e precisão G4 |
| M8 Notebook e T4 | Notebook da aula fracionado por caixa e reprodução na T4 | Gerador do notebook, notebook, checkpoints da aula, TOST | [PROBLEMA](PROBLEMA.md), [ARQUITETURA](ARQUITETURA.md) | Precisão G3, orçamento G6, corpus G4 e G5, tokenizador G3 e G4 |
| M9 Leitura | Níveis de coerência medidos | Lotes cegos, respostas, concordância, intervalo de épocas por nível | [decisão de orçamento](decisao_orcamento.md), [SPEC](SPEC.md) | Orçamento G7 |
| M10 Etapa 2 | Notebook didático de tokenizador e bancada comparativa | Notebook de treino e avaliação de tokenizador, varredura de $V$ e regex para português, proposta do autor em `privado/` | [PROBLEMA](PROBLEMA.md), [decisão do tokenizador](decisao_tokenizador.md) | Plano da etapa 2 aprovado pelo autor e bancada medida |
| M11 Etapa 3 | Pós-treino por preferência (DPO) nos dois modos e seu efeito sobre coerência e calibração | Decisão da fonte de recompensa por debate, pares de preferência, DPO com NELBO na difusão, medidas antes e depois | [PROBLEMA](PROBLEMA.md) | Plano da etapa 3 aprovado pelo autor e medidas antes e depois registradas |
| M12 Artigo tutorial | Artigo no estilo tutorial para o arXiv, sobre as etapas 1 a 3 | Manuscrito, figuras da SPEC, revisão adversarial (skill `reviewer2-guardian`), versão aprovada pelo autor | Resultados de M7, M9, M10 e M11 | Aprovação do autor |
| M13 Publicação | Repositório público e citável | Verificação de publicabilidade, repositório no GitHub, lançamento no Zenodo com DOI, README com selo e citação, submissão ao arXiv | [PROBLEMA](PROBLEMA.md), skill `zenodo-publish` | DOI ativo, README atualizado e artigo submetido |

## Dependências

```
M1 ──► M2 ──► M3 ──► M4 ──► M5 ──► M6 ──► M7 ──► M9 ──► M10 ──► M11 ──► M12 ──► M13
                      │                    │
                      └────────► M8 ◄──────┘
                                 │
                                 └──► M12 (o notebook entra no artigo)
```

O notebook (M8) começa assim que o modelo passa em CPU (M4), porque as primeiras seções só dependem das caixas 0 a 5. As seções de comparação e de leitura entram depois de M7 e M9. A publicação no GitHub, no Zenodo e no arXiv exige autorização explícita do autor no momento do envio. A proposta de tokenizador do autor, mantida em `privado/`, só entra no repositório público ou no artigo por decisão expressa dele.

## Critério de pronto de cada marco

Um marco está pronto quando os portões que o encerram estão aprovados, os testes passam a partir de um clone limpo e o [cookbook](../cookbook/README.md) tem a receita do marco.

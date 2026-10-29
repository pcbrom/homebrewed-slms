# Relatório de construção dos corpora

Este relatório registra os números dos portões G1 e G3 da [decisão de corpus](../docs/decisao_corpus.md), medidos sobre os arquivos de `corpus/A/` e `corpus/B/` gerados por `scripts/construir_corpus_a.py` e `scripts/construir_corpus_b.py`. Bytes são sempre bytes UTF-8 do texto limpo em NFC, e MB vale $10^6$ bytes. O portão G0, que mede tokens, depende do tokenizador da partição de treino e fica fora deste relatório. O portão G2 depende da auditoria manual em `B/auditoria_classes.csv`, ainda por preencher.

## Corpus A

### Composição e limpeza

O corpus tem as 10 obras em prosa de Machado de Assis previstas na decisão. *Poesias completas* (61653) e *Relíquias de casa velha* (67935) foram baixadas, registradas no manifesto com o motivo da exclusão e deixadas fora. O texto de cada obra é o trecho entre as marcas `START` e `END` do Gutenberg, com CRLF convertido em LF e normalização NFC. Tudo o que precede o primeiro capítulo sai como folha de rosto, o que inclui índice, lista de obras do autor, advertências e prefácios. O corpo termina na primeira marca de fim (`FIM`, `FIN` ou `INDICE`).

Dentro de cada capítulo, as linhas de um parágrafo são unidas e os parágrafos ficam separados por uma linha em branco. Parágrafos com todas as linhas recuadas, que são versos ou cartas destacadas, mantêm as quebras de linha. O sublinhado de itálico sai, as notas entre colchetes (marcadores como `[1]` e notas de transcrição como `[signature]`) saem, e os asteriscos isolados ou duplos, que marcam correções de transcrição, também. Os três asteriscos que ocultam um nome no original ficam. O travessão da transcrição, escrito com hífens, fica padronizado como `--`. A grafia original é mantida.

### Regra de corte por obra

| Obra | Ebook | Regra | Unidades |
|---|---|---|---|
| Dom Casmurro | 55752 | Algarismo romano isolado | 148 |
| Memórias póstumas de Brás Cubas | 54829 | `CAPITULO` mais romano, inclusive a grafia `CAPIULO` do capítulo CXXIV na fonte | 160 |
| Quincas Borba | 55682 | `CAPITULO` mais romano ou `PRIMEIRO` | 201 |
| Papeis avulsos | 57001 | Conto: título em maiúsculas após quatro linhas em branco | 12 |
| Historias sem data | 33056 | Conto, como acima | 18 |
| A mão e a luva | 53101 | Algarismo romano isolado | 19 |
| Helena | 67162 | `CAPITULO` mais romano | 28 |
| Esaú e Jacob | 56737 | `CAPITULO` mais romano ou `PRIMEIRO` | 120 |
| Memorial de Ayres | 55797 | Entrada de data em início de linha, como "9 de Janeiro.", e "Sem data." | 165 |
| Yayá Garcia | 67780 | Algarismo romano isolado | 17 |

As duas coletâneas usam o conto como unidade, porque não têm numeração única de capítulos. Nas obras de capítulo, a numeração foi conferida como sequência completa de 1 a $n$, sem lacuna nem repetição. A edição de *Esaú e Jacob* no Gutenberg termina no capítulo CXX. No *Memorial*, as datas dentro de cartas, que vêm recuadas, não abrem entrada, e as linhas de ano (1888, 1889) saem.

### Portão G1

A divisão segue a decisão. Em cada obra, na ordem de leitura, vêm treino, um capítulo de margem, o bloco de validação, um capítulo de margem e o bloco de teste, que fica no fim da obra. Cada bloco é a menor sequência de capítulos inteiros com pelo menos 5 capítulos e 30.720 bytes (30 KiB, o que cumpre também a leitura de 30 KB como 30.000 bytes). O treino de cada obra precisa atingir o mesmo mínimo. A obra que não comporta os dois blocos sai do teste e fica com validação.

Na tabela, cada célula traz capítulos e bytes.

| Obra | Treino | Margem | Validação | Teste |
|---|---|---|---|---|
| Dom Casmurro | 113 / 310.744 | 2 / 3.755 | 16 / 31.949 | 17 / 35.136 |
| Memórias póstumas de Brás Cubas | 116 / 289.236 | 2 / 9.883 | 22 / 31.394 | 20 / 31.454 |
| Quincas Borba | 164 / 386.166 | 2 / 7.065 | 13 / 30.814 | 22 / 33.199 |
| Papeis avulsos | 6 / 213.986 | 1 / 21.423 | 5 / 88.966 | fora do teste |
| Historias sem data | 6 / 93.352 | 2 / 45.319 | 5 / 84.750 | 5 / 77.699 |
| A mão e a luva | 7 / 72.443 | 2 / 19.662 | 5 / 59.953 | 5 / 53.039 |
| Helena | 16 / 194.861 | 2 / 19.688 | 5 / 58.441 | 5 / 68.576 |
| Esaú e Jacob | 96 / 348.796 | 2 / 4.128 | 10 / 31.205 | 12 / 31.544 |
| Memorial de Ayres | 121 / 222.789 | 2 / 2.669 | 23 / 31.000 | 19 / 31.429 |
| Yayá Garcia | 5 / 88.999 | 2 / 43.198 | 5 / 91.100 | 5 / 112.751 |
| Total | 650 / 2.221.372 | 19 / 176.790 | 109 / 539.572 | 110 / 474.827 |

**Resultado: aprovado.** As 9 obras do teste cumprem o bloco mínimo nas três partições. *Papeis avulsos* sai do teste, porque 12 contos não comportam dois blocos de 5 contos, duas margens e um treino de 5 contos. A exclusão está registrada na decisão de corpus. O corpus A soma 3,41 MB, com 2,22 MB de treino.

As listas de nomes próprios, em `A/nomes/<obra>.txt`, saem só dos capítulos de treino de cada obra e somam 623 entradas. Entra a palavra capitalizada fora do início de frase com pelo menos 2 ocorrências, cuja forma em minúsculas aparece no máximo um quarto das vezes. Meses, dias da semana e abreviaturas de tratamento ficam de fora. A heurística aceita também substantivos que a grafia da época capitaliza, como "Deus" e "Excellencia", o que a lista versionada deixa visível.

## Corpus B

### Lista e classes

As listas do Senado de 2019 a 2022 vieram completas, sem resposta 503, com 728 leis ordinárias numeradas (tipo `LEI-n`): 185 em 2019, 143 em 2020, 172 em 2021 e 228 em 2022. A regra de classe é exclusiva e aplicada na ordem crédito, orçamento, homenagem, alteradora e substantiva.

| Classe | Leis | Fração da lista |
|---|---|---|
| Crédito | 168 | 23,1% |
| Orçamento | 22 | 3,0% |
| Homenagem | 94 | 12,9% |
| Alteradora | 241 | 33,1% |
| Substantiva | 203 | 27,9% |

A classe de orçamento inclui as leis que alteram a lei de diretrizes orçamentárias ou o plano plurianual, pois a ordem da regra a examina antes da alteradora. Por isso ela fica acima do 1,1% medido na amostra da decisão. A classe de homenagem cobre denominações, títulos honoríficos, patronos, datas comemorativas e meses de conscientização. A alteradora cobre as ementas que começam por "Altera", "Acrescenta", "Revoga", "Dá nova redação", "Confere nova redação" e verbos afins.

### Portão G3

| Critério | Medido | Limite | Resultado |
|---|---|---|---|
| Páginas obtidas | 444 de 444 (100%) | pelo menos 99% | aprovado |
| Volume | 4,76 MB em 441 leis | pelo menos 2,5 MB | aprovado |
| Maior grupo | 5,78% dos bytes (Lei 14.133, licitações, documento único) | no máximo 10% | aprovado |

Todas as 444 páginas das classes substantiva e alteradora responderam no padrão `_ato2019-2022/{ano}/lei/l{n}.htm`. Três leis (13.871, 14.016 e 14.125) saem do corpus por estarem integralmente revogadas, sem texto vigente na página compilada, o que deixa 441 leis com texto (99,3% das páginas): 240 alteradoras e 201 substantivas.

A limpeza remove o texto riscado, marcado por `<strike>`, `<del>` ou estilo `line-through`, as anotações editoriais entre parênteses ("Redação dada", "Incluído pela", "Vide", "Vigência", "Revogado pela" e afins), os rótulos de navegação da página e o rodapé a partir de "Este texto não substitui o publicado". O texto começa no título da lei, que é conferido contra o número esperado.

### Grupos e divisão

O agrupamento une cada lei à primeira lei citada na ementa e, por MinHash de 5-gramas de palavras com dígitos normalizados (128 permutações, 32 bandas, confirmação pelo Jaccard exato), une as quase-cópias com Jaccard de pelo menos 0,8. Um par passou do limiar, as Leis 14.160 e 14.458, com Jaccard de 0,811. Resultam 297 grupos, dos quais 234 têm uma única lei. A partição de cada grupo vem do sha256 do identificador do grupo módulo 100.

| Partição | Leis | Grupos | Bytes |
|---|---|---|---|
| Treino | 401 | 269 | 4.312.670 |
| Validação | 21 | 16 | 173.567 |
| Teste | 19 | 12 | 274.481 |

O texto dentro de citação (`<blockquote>`), que nas leis alteradoras reproduz a redação nova de outra lei, soma 39,6% dos bytes do corpus. O maior grupo do teste é o da Lei 10.406 (Código Civil), com 7 leis e 53,2% dos bytes do teste.

## Limitações

- O treino do A ficou com 2,22 MB, abaixo dos cerca de 3,2 MB estimados antes da construção. O bloco mínimo de 5 capítulos consome boa parte das obras de capítulos longos, e em *Yayá Garcia* o teste tem mais bytes que o treino.
- O B tem 4,76 MB contra 2,22 MB de treino do A, de modo que a comparação entre domínios não fica pareada em volume. A decisão não fixa portão de paridade de volume.
- Validação e teste do B são pequenos (21 e 19 leis), porque a partição por grupo sorteia 10% dos grupos, e não dos bytes.
- Itens revogados cujo texto estava riscado deixam marcas residuais como "I -;", e as assinaturas de autoridades permanecem por fazerem parte do ato.
- O Planalto reescreve as páginas compiladas, e a coleta ocorreu em 22 e 23/10/2026, datas registradas por lei no manifesto.
- O agrupamento não captura leis que alteram outras só no corpo do texto, conforme os riscos declarados na decisão.

## Reprodução

```bash
python3 scripts/construir_corpus_a.py
python3 scripts/construir_corpus_b.py
python3 scripts/auditoria_classes.py
python3 -m pytest -q tests/test_dados.py
```

Os scripts usam só a biblioteca padrão, guardam os arquivos baixados em `dados_brutos/`, fora do versionamento, e reaproveitam o que já existe lá. O hash de `A/documentos.jsonl` é `364aa78f417a13fd49a71a380cdff9d86ffee8c3ef3a159237fe50befcc43f77` e o de `B/documentos.jsonl` é `c51d3d40ca60bfeb318408e4bba0732a87d87b97da2a44d30d0b68943086ec71`.

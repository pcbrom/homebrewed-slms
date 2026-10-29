# Construir os corpora A e B

Os dois corpora limpos já estão versionados em `corpus/A` e `corpus/B`, e o notebook não baixa nada. Esta receita reconstrói os dois a partir das fontes, conforme a [decisão de corpus](../docs/decisao_corpus.md). Os scripts usam só a biblioteca padrão do Python, guardam os arquivos baixados em `dados_brutos/`, fora do git, e os reaproveitam nas execuções seguintes.

```bash
python3 scripts/construir_corpus_a.py
python3 scripts/construir_corpus_b.py
```

O corpus A baixa 12 arquivos do Project Gutenberg (cerca de 4 MB). O corpus B baixa as listas de leis do Senado de 2019 a 2022 e cerca de 440 páginas do Planalto, com pausa de 1 segundo entre requisições, o que leva cerca de 15 minutos na primeira vez. A opção `--forcar` baixa tudo de novo.

## Conferir

```bash
sha256sum corpus/A/documentos.jsonl corpus/B/documentos.jsonl
.venv/bin/python -m pytest -q tests/test_dados.py
```

O hash de `documentos.jsonl` também está em `divisao.json`. O relatório da construção, com documentos e bytes por obra e partição e os números dos portões G1 e G3, está em [corpus/RELATORIO_CONSTRUCAO.md](../corpus/RELATORIO_CONSTRUCAO.md). As páginas compiladas do Planalto mudam com o tempo, e uma reconstrução posterior do B pode diferir da versionada. A data de coleta de cada lei está no manifesto.

## Auditoria da regra de classe (portão G2)

```bash
python3 scripts/auditoria_classes.py
```

O script sorteia, com semente fixa, 72 ementas da lista completa e grava `corpus/B/auditoria_classes.csv`. A coluna `classe_correta` é preenchida à mão. A regra passa com zero erros em 72, o que põe o limite superior de Clopper-Pearson de 95% abaixo de 5%.

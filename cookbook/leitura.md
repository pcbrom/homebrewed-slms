# Montar e analisar a leitura cega

Esta receita reproduz o marco M9, conforme a [decisão de orçamento](../docs/decisao_orcamento.md), a seção 10 da [SPEC](../docs/SPEC.md) e a [rubrica dos avaliadores](../docs/RUBRICA_LEITURA.md). Ela depende das corridas da grade descritiva do M7, listadas em `preregistro/m7_grade.json` com `run_id`, `modo`, `ponto` e `semente` de cada uma. A rubrica, o comprimento dos itens e o limiar de cópia entram no pré-registro antes de qualquer lote ir para leitura.

```bash
.venv/bin/python scripts/m9.py amostras --grade preregistro/m7_grade.json --dispositivo cuda
.venv/bin/python scripts/m9.py controles
.venv/bin/python scripts/m9.py lotes
```

A etapa `amostras` gera 8 amostras por corrida, corta cada uma no primeiro `[EOS]` e recorta 60 palavras a partir de uma posição sorteada. A etapa `controles` sorteia, da partição de teste, 500 excertos humanos do mesmo comprimento para o limiar de cópia $k$ e para a referência de N1, e mais 20 pares de controle por lote, cada um formado por um excerto e a versão dele com as palavras embaralhadas. As amostras do modelo começam num início de documento e os excertos humanos no meio de um, então o recorte de posição sorteada nos dois impede que a primeira frase revele a origem do item. A etapa `lotes` descarta as amostras com trecho copiado maior que $k$, escolhe 12 itens por modo em cada ponto, igualmente entre as sementes, e grava em `leitura/` um `lote_<n>.csv` e uma `chave_<n>.json` por ponto, além de `n1_copia.json` com N1 e a taxa de descarte por cópia, com intervalo de Clopper-Pearson.

Os avaliadores recebem só os `lote_<n>.csv` e a rubrica, e devolvem um `respostas_<avaliador>_<n>.csv` por lote. As chaves ficam fora do alcance deles até o fim da leitura e só são commitadas depois. Com as respostas em `leitura/`,

```bash
.venv/bin/python scripts/m9.py analisar --autor <nome do autor>
```

filtra os avaliadores pelos pares de controle, calcula o alfa de Krippendorff ordinal por nível com e sem o autor e decide, por ponto e modo, se cada nível de N2 a N4 foi atingido, gravando `leitura/resultado.json`.

## Testes

```bash
.venv/bin/python -m pytest -q tests/test_m9.py tests/test_leitura.py
```

# Treinar e testar o tokenizador

O tokenizador do experimento já está versionado em `tokenizador/<corpus>/`, um por corpus. Esta receita o reconstrói a partir do corpus, conforme a [decisão do tokenizador](../docs/decisao_tokenizador.md).

```bash
.venv/bin/python scripts/treinar_tokenizador.py --corpus A
.venv/bin/python scripts/treinar_tokenizador.py --corpus B
```

O script treina a biblioteca `tokenizers` para cada valor da grade $\{1.024, 2.048, 4.096, 8.192\}$ na partição de treino, mede a fração de tokens aprendidos ausentes da validação e escolhe o maior $V$ com fração de até 10%. Depois treina o BPE didático numa amostra de 440 KB do treino e confere que as fusões e os ids no retido são idênticos aos da biblioteca. Se divergirem, o script sai com erro, que é o portão G4. O resultado fica em `tokenizador/<corpus>/tokenizer.json` e `tokenizador/<corpus>/metadados.json`. O treino leva cerca de 1 minuto em CPU. A regra escolhe $V = 4.096$ no corpus A e $V = 1.024$ no corpus B, cujo retido tem 10,9% de tokens ausentes já em $V = 2.048$.

## Testes do contrato

```bash
.venv/bin/python -m pytest -q tests/test_tokenizador.py
```

Os testes U1 a U6 valem para qualquer tokenizador que cumpra o contrato, inclusive um trivial de bytes. Os dois testes específicos de BPE conferem a equivalência entre a implementação didática e a biblioteca.

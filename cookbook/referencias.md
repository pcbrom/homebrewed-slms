# Referências de acesso aberto

Toda fonte científica citada no projeto é de acesso aberto e fica registrada em `referencias/fontes.json`. O script `scripts/referencias.py` baixa cada uma, fixa a versão, lê a licença e confere o sha256.

## Baixar e conferir tudo

```bash
python3 scripts/referencias.py
```

O script usa só a biblioteca padrão do Python, respeita a pausa de 3 segundos entre requisições ao arXiv e regenera `referencias/REFERENCIAS.md`. Os PDFs com licença Creative Commons ou de domínio público ficam em `referencias/pdf/`, versionados. Os demais vão para `referencias/pdf_local/`, fora do git.

## Só conferir os hashes

```bash
python3 scripts/referencias.py --verificar
```

Sai com código 1 se algum arquivo divergir do hash registrado em `referencias/trava.json`.

## Acrescentar uma fonte

1. Incluir uma entrada em `referencias/fontes.json`, com `chave`, `arxiv` ou `url_pdf` e `citado_em`. Para fonte sem arXiv, informar também `titulo`, `autores`, `ano`, `veiculo`, `versao` e `licenca`.
2. Rodar `python3 scripts/referencias.py`.
3. Conferir em `referencias/REFERENCIAS.md` a licença e a citação em ABNT.

O ano da referência é o do primeiro depósito no arXiv, estável entre revisões. A versão baixada fica indicada na referência. Para artigos baixados da ACL Anthology, vale o ano da publicação.

# Rodar os portões em CPU

Os portões G0 e G1 da [decisão de formulação](../docs/decisao_formulacao.md) são bloqueantes e rodam sobre o código do pacote, em float64:

```bash
.venv/bin/python -m pytest -q tests/test_portoes_formulacao.py
```

| Teste | O que verifica |
|---|---|
| `test_g0_nelbo_igual_media_das_ordens` | A NELBO por número de máscaras é igual à média, sobre as $L!$ ordens de revelação, de $-\log \prod p$ (Ou et al., 2024, Teorema 2), com tolerância de $10^{-9}$ |
| `test_g0_normalizacao_e_limite` | Cada ordem define uma distribuição normalizada, e a NELBO é maior ou igual a $-\log \mathbb{E}_\sigma[\prod p]$ |
| `test_g0_defeito_logit_de_mask_reprova` | Uma rede que não bloqueia o logit de `[MASK]` quebra a normalização |
| `test_g0_defeito_alvo_visivel_reprova` | Uma rede que vê o próprio alvo quebra a normalização |
| `test_g1_preenchimento_fecha_o_vazamento_de_comprimento` | Com condicionais exatas, a NLL autoregressiva e a NELBO coincidem quando o preenchimento é previsto, e dar o comprimento à difusão reduz a perda dela exatamente na entropia do comprimento |
| `test_g1_soma_de_bytes_do_iterador` | A avaliação soma os bytes de todos os documentos |

Os testes do contrato do tokenizador (U1 a U6 e a equivalência entre o BPE didático e a biblioteca) estão em `tests/test_tokenizador.py`.

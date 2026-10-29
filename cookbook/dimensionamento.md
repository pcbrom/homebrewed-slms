# Dimensionar o treino ao hardware

O script `scripts/dimensionar.py` estima vazão, tempo, volume de dados e memória de treino para qualquer GPU, com as fórmulas de [docs/DIMENSIONAMENTO.md](../docs/DIMENSIONAMENTO.md). Usa só a biblioteca padrão do Python.

## Uso

```bash
python3 scripts/dimensionar.py --pico-tflops 65 --vram-gb 16 --arq 4x256 6x384 8x512 --horas 1
```

| Argumento | Significado |
|---|---|
| `--pico-tflops` | Pico denso dos núcleos tensoriais na precisão do treino, pela especificação do fabricante |
| `--vram-gb` | Memória da GPU |
| `--mfu` | Fração do pico aproveitada. Padrão 0,3 e 0,5. Trocar pelo valor medido |
| `--arq` | Arquiteturas no formato camadas x dimensão |
| `--ctx`, `--vocab` | Comprimento de contexto e tamanho do vocabulário |
| `--tokens-por-param` | Alvo de dados por parâmetro, padrão 20 |
| `--sem-embedding` | Conta só os parâmetros fora do embedding no alvo de dados |
| `--horas` | Orçamento de tempo, para obter o tamanho ótimo que cabe nele |

## Interpretar

A coluna "extra" mostra quanto a atenção e a camada de saída acrescentam à aproximação $6N$. Em modelos pequenos esse acréscimo passa de 30%. A coluna de memória mostra a fração da VRAM ocupada só por pesos e estado do AdamW, antes das ativações. As ativações e o MFU real se medem com o procedimento da seção 6 do documento de dimensionamento.

<< 'EOF'
# llm-tattoo-vlm-benchmark

Repositório do projeto de Deep Learning: benchmark de Vision-Language Models (VLMs) no cenário **open-set** do dataset **TSSD2023** (tattoo semantic segmentation), comparando:

1) **Baseline**: imagem inteira  
2) **Crop GT (fundo preto)**: recorte usando máscara GT + background preto  
3) **Crop GT (fundo branco)**: recorte usando máscara GT + background branco  

## Objetivo
Isolar e medir o componente **semântico** (rotulagem textual de classes) das VLMs quando a segmentação é controlada por GT (“segmentação oráculo”), evitando comparar diretamente com métricas pixel-a-pixel de segmentadores.

## Modelos avaliados
- Gemma3
- Qwen2.5-VL
- LLaMA 3.2 Vision

## Métrica principal: micro-F1 (multi-label por imagem)
Para cada imagem:
- `GT` = conjunto de classes verdadeiras
- `Pred` = conjunto de classes previstas (após parsing/mapeamento para vocabulário controlado)

Definições:
- `TP = |GT ∩ Pred|`
- `FP = |Pred \\ GT|`
- `FN = |GT \\ Pred|`

No dataset:
`micro-F1 = 2TP / (2TP + FP + FN)`

## Estrutura do repositório
- `run_experiments.py`: orquestração geral
- `experiments/`: scripts de geração de crops, execução das VLMs, parsing, métricas, agregações e geração de figuras/tabelas
- `data_meta/`: metadados pequenos (ex.: `tssd2023_id2name.json`)
- `environment.yml`: ambiente (conda/mamba)

## Reprodutibilidade
### Criar ambiente
```bash
cd ~/projects/llm-tattoo
mamba env create -f environment.yml
mamba activate llm-tattoo

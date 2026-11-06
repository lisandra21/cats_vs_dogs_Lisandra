# AlexNet Binary Image Classification: Cats vs Dogs

Repositório contendo a implementação e avaliação de uma Rede Neural Convolucional inspirada na arquitetura AlexNet (Krizhevsky et al., 2012), adaptada para a classificação binária de imagens (Gatos vs. Cachorros) em PyTorch.

## Visão Geral

- **Tarefa:** Classificação binária (0 = Gato, 1 = Cachorro).
- **Framework:** PyTorch & Torchvision.
- **Dataset:** `bhavikjikadara/dog-and-cat-classification-dataset` (divisão aleatória: 80% treino, 10% validação, 10% teste via `train_test_split`).
- **Dimensão de Entrada:** Imagens RGB de 224 × 224 × 3.

## Arquitetura da Rede (`AlexNetBinary`)

A rede mantém a estrutura de 5 camadas convolucionais da AlexNet seguidas por 3 camadas totalmente conectadas, com adaptação final para saída escalar probabilística:

- **Conv 1:** 96 filtros 11 × 11, stride 4, padding 2 → ReLU → MaxPool (3 × 3, stride 2) → Saída: 27 × 27 × 96
- **Conv 2:** 256 filtros 5 × 5, stride 1, padding 2 → ReLU → MaxPool (3 × 3, stride 2) → Saída: 13 × 13 × 256
- **Conv 3:** 384 filtros 3 × 3, stride 1, padding 1 → ReLU → Saída: 13 × 13 × 384
- **Conv 4:** 384 filtros 3 × 3, stride 1, padding 1 → ReLU → Saída: 13 × 13 × 384
- **Conv 5:** 256 filtros 3 × 3, stride 1, padding 1 → ReLU → MaxPool (3 × 3, stride 2) → Saída: 6 × 6 × 256
- **Vetorização (Flatten):** 6 × 6 × 256 = 9216 elementos.
- **Classificador (FC):**
  - **FC 1:** Linear (9216 → 4096) + ReLU
  - **FC 2:** Linear (4096 → 4096) + ReLU
  - **FC 3:** Linear (4096 → 1000) + ReLU
  - **Saída:** Linear (1000 → 1) + Sigmoide

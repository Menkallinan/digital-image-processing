# 🍎 Projeto de Processamento Digital de Imagens — Classificação de Frutas

## 🧠 Visão Geral

Este projeto faz parte da disciplina **Processamento Digital de Imagens (PDI)** da **Universidade Tecnológica Federal do Paraná (UTFPR)**.  
O objetivo é aplicar técnicas de **análise, segmentação, normalização e classificação de imagens** em um conjunto de dados contendo **diferentes tipos de frutas**, explorando todo o pipeline de processamento — desde a aquisição até o aprendizado de máquina.

O trabalho foi desenvolvido **em equipe**, com ênfase em **comunicação, colaboração e divisão eficiente de tarefas**, simulando um ambiente de projeto profissional.

---

## 🧩 Estrutura do Projeto

O projeto foi dividido em **6 partes principais**, cada uma representando uma etapa do fluxo de trabalho completo:

### 🥭 **Parte 1 — Base de Dados das Frutas**
- Contém as **imagens originais** das frutas em formato `.png`.
- Cada classe representa uma fruta diferente, totalizando **10 categorias**.
- As imagens servem como base para todas as etapas subsequentes.

### 🍋 **Parte 2 — Bounding Box**
- As imagens são processadas para **delimitar as frutas com caixas (bounding boxes)**.
- Utiliza anotações JSON para localizar os objetos e gerar visualizações com contornos verdes.
- Notebook: `bounding_box_GT_fruits_.ipynb`.

### 🍓 **Parte 3 — Data Augmentation**
- Amplia a base de dados por meio de transformações geométricas e radiométricas:
  - Conversão logarítmica;
  - Potenciação;
  - Convolução com filtro da média.
- Aumenta a robustez do modelo de classificação.
- Notebook: `Fruit_Image_Montage_Visualization.ipynb`.

### 🍈 **Parte 4 — Normalização**
- As imagens são **normalizadas** (intensidades reescaladas) para garantir uniformidade no treinamento.
- Resulta em um dataset consistente e balanceado.
- Notebook: `normalized_dataset.ipynb`.

### 🍑 **Parte 5 — Ground Truth**
- Armazena as versões **binárias das imagens** (máscaras) que representam as regiões de interesse (ROI).
- Essencial para comparar resultados de segmentação e treinamento supervisionado.

### 🍌 **Parte 6 — Classificador**
- Implementa um **modelo de aprendizado de máquina** capaz de **reconhecer o tipo de fruta**.
- Utiliza técnicas de classificação supervisionada e métricas de desempenho (acurácia, matriz de confusão, etc.).
- Notebook: `Classifier.ipynb`.

---

## ⚙️ Tecnologias Utilizadas

- **Python 3.x**
- **OpenCV (cv2)** — Processamento de imagens  
- **scikit-image** — Filtros, transformações e montagens  
- **NumPy** — Operações matriciais  
- **Matplotlib** — Visualização de imagens e resultados  
- **scikit-learn** — Classificação e métricas  
- **Pandas** — Manipulação de dados  
- **Google Colab** — Ambiente de execução e testes

---

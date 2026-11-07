# Classificação da qualidade de vinhos

Tech Challenge da Fase 2 da Pós Tech FIAP (Data Analytics).

O objetivo é prever se um vinho tinto é de alta qualidade (nota maior ou igual a 7) ou de baixa/média qualidade (nota menor que 7) a partir de 11 medições físico-químicas, e entender quais delas mais influenciam a qualidade.

## Dados

Usamos o [Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset) do Kaggle (`data/WineQT.csv`). São 1.143 vinhos tintos, dos quais 125 eram duplicados. Depois de remover as duplicadas ficamos com 1.018 vinhos, sendo 137 (13,5%) de alta qualidade.

## Estrutura

```
wine-quality-classification/
├── data/              base de dados
├── notebooks/         notebook com a análise e os modelos
├── src/               funções de pré-processamento e modelagem
├── results/           métricas (csv) e gráficos gerados pelo notebook
├── apresentacao/      apresentação executiva (pptx e pdf)
├── relatorio/         relatório técnico (pdf)
├── requirements.txt
└── README.md
```

## O que foi feito

1. Transformamos a nota em duas classes (corte em 7).
2. Na análise exploratória olhamos distribuições, outliers, valores inconsistentes, balanceamento das classes e correlações.
3. No pré-processamento removemos duplicadas, criamos três variáveis novas (acidez total, proporção de acidez volátil e proporção de SO2 livre), separamos treino e teste de forma estratificada (80/20) e deixamos o escalonamento dentro de um Pipeline para não ter vazamento de dados.
4. Treinamos KNN (escolhendo o k e o tipo de escalonamento com validação cruzada), Regressão Logística e Random Forest.
5. Comparamos os modelos com F1, recall, precisão, ROC-AUC, PR-AUC e matriz de confusão. A acurácia não serve sozinha aqui porque as classes são desbalanceadas.
6. Analisamos a importância das variáveis no Random Forest e os coeficientes da Regressão Logística.

## Resultados no teste

204 vinhos, sendo 27 de alta qualidade.

| Modelo | Acurácia | Precisão | Recall | F1 | ROC-AUC | F1 na validação cruzada |
|---|---|---|---|---|---|---|
| Baseline (sempre "baixa/média") | 0,868 | 0,000 | 0,000 | 0,000 | 0,500 | 0,000 |
| KNN (k=7, Min-Max) | 0,912 | 0,737 | 0,519 | 0,609 | 0,865 | 0,493 |
| Regressão Logística | 0,789 | 0,357 | 0,741 | 0,482 | 0,886 | 0,525 |
| Random Forest | 0,824 | 0,408 | 0,741 | 0,526 | 0,922 | 0,551 |

Escolhemos o Random Forest, que teve o melhor F1 na validação cruzada e o melhor ROC-AUC no teste. Ele encontrou 20 dos 27 vinhos de alta qualidade. O KNN dá menos falsos alarmes, mas deixa passar metade dos vinhos bons.

As variáveis que mais influenciam a qualidade são o álcool (quanto mais, melhor), os sulfatos (quanto mais, melhor) e a acidez volátil (quanto menos, melhor).

## Como executar

No Google Colab: suba a pasta `wine-quality-classification` para o "Meu Drive", abra o notebook e rode todas as células. Se a pasta estiver em outro lugar, ajuste o caminho na primeira célula.

Localmente:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/wine_quality_classification.ipynb
```

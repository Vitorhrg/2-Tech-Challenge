"""Funções de carregamento e preparação dos dados."""
import io
import os
import urllib.request
import zipfile

import pandas as pd

KAGGLE_URL = 'https://www.kaggle.com/api/v1/datasets/download/yasserh/wine-quality-dataset'
QUALITY_THRESHOLD = 7

FEATURES = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
            'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

ENGINEERED_FEATURES = ['total acidity', 'volatile acidity ratio', 'free so2 ratio']

FEATURE_LABELS = {
    'fixed acidity': 'Acidez fixa (g/L)',
    'volatile acidity': 'Acidez volátil (g/L)',
    'citric acid': 'Ácido cítrico (g/L)',
    'residual sugar': 'Açúcar residual (g/L)',
    'chlorides': 'Cloretos (g/L)',
    'free sulfur dioxide': 'SO2 livre (mg/L)',
    'total sulfur dioxide': 'SO2 total (mg/L)',
    'density': 'Densidade (g/cm³)',
    'pH': 'pH',
    'sulphates': 'Sulfatos (g/L)',
    'alcohol': 'Álcool (% vol.)',
}


def load_data(data_dir='data'):
    """Lê o WineQT.csv. Se o arquivo não estiver na pasta, baixa do Kaggle."""
    path = os.path.join(data_dir, 'WineQT.csv')
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        req = urllib.request.Request(KAGGLE_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as resp:
            zipfile.ZipFile(io.BytesIO(resp.read())).extract('WineQT.csv', data_dir)
    return pd.read_csv(path)


def remove_duplicates(df):
    """Remove vinhos repetidos (mesmas medições e mesma nota), ignorando o Id."""
    return df.drop_duplicates(subset=FEATURES + ['quality']).reset_index(drop=True)


def create_target(df, threshold=QUALITY_THRESHOLD):
    """Cria a coluna high_quality: 1 se a nota for >= threshold, senão 0."""
    df = df.copy()
    df['high_quality'] = (df['quality'] >= threshold).astype(int)
    return df


def add_engineered_features(df):
    """Cria as novas variáveis. São calculadas linha a linha, sem usar estatísticas da base."""
    df = df.copy()
    df['total acidity'] = df['fixed acidity'] + df['volatile acidity'] + df['citric acid']
    df['volatile acidity ratio'] = df['volatile acidity'] / df['total acidity']
    df['free so2 ratio'] = df['free sulfur dioxide'] / df['total sulfur dioxide']
    return df


def check_inconsistencies(df):
    """Conta quantos vinhos violam regras físico-químicas básicas."""
    rules = {
        'valores negativos': (df[FEATURES] < 0).any(axis=1),
        'SO2 livre maior que SO2 total': df['free sulfur dioxide'] > df['total sulfur dioxide'],
        'ácido cítrico maior que acidez fixa': df['citric acid'] > df['fixed acidity'],
        'pH fora de 2,5 a 4,5': ~df['pH'].between(2.5, 4.5),
        'densidade fora de 0,98 a 1,01': ~df['density'].between(0.98, 1.01),
        'álcool fora de 5% a 16%': ~df['alcohol'].between(5, 16),
        'nota fora de 0 a 10': ~df['quality'].between(0, 10),
    }
    return pd.DataFrame({'regra': list(rules), 'ocorrencias': [int(m.sum()) for m in rules.values()]})


def iqr_outliers(df, columns=FEATURES, k=1.5):
    """Quantidade de outliers por variável pelo critério do IQR."""
    rows = []
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        low, high = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
        n = ((df[col] < low) | (df[col] > high)).sum()
        rows.append({'variavel': col, 'limite_inferior': low, 'limite_superior': high,
                     'outliers': n, 'percentual': n / len(df) * 100})
    return pd.DataFrame(rows).sort_values('outliers', ascending=False).reset_index(drop=True)

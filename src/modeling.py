"""Pipelines dos modelos, validação cruzada e métricas.

O escalonamento fica dentro do Pipeline para ser ajustado só com os dados de treino
(em cada fold da validação cruzada e no treino final), evitando vazamento de dados.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             f1_score, make_scorer, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

RANDOM_STATE = 42

SCALERS = {'MinMax': MinMaxScaler, 'Standard': StandardScaler}

# zero_division=0 evita aviso quando um modelo não prevê nenhum vinho como alta qualidade
SCORING = {
    'accuracy': 'accuracy',
    'balanced_accuracy': 'balanced_accuracy',
    'precision': make_scorer(precision_score, zero_division=0),
    'recall': 'recall',
    'f1': make_scorer(f1_score, zero_division=0),
    'roc_auc': 'roc_auc',
    'average_precision': 'average_precision',
}

METRIC_NAMES = {
    'accuracy': 'Acurácia',
    'balanced_accuracy': 'Acurácia balanceada',
    'precision': 'Precisão',
    'recall': 'Recall',
    'f1': 'F1',
    'roc_auc': 'ROC-AUC',
    'average_precision': 'PR-AUC',
}


def get_cv(n_splits=5):
    """Validação cruzada estratificada (cada fold mantém a proporção das classes)."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def knn_pipeline(k=5, scaler='Standard'):
    return Pipeline([('scaler', SCALERS[scaler]()),
                     ('model', KNeighborsClassifier(n_neighbors=k))])


def logreg_pipeline(C=1.0):
    return Pipeline([('scaler', StandardScaler()),
                     ('model', LogisticRegression(C=C, class_weight='balanced', max_iter=5000))])


def rf_pipeline(**params):
    return Pipeline([('model', RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                                      random_state=RANDOM_STATE, n_jobs=-1, **params))])


def cv_summary(model, X, y, cv):
    """Média e desvio padrão de cada métrica na validação cruzada."""
    scores = cross_validate(model, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    result = {}
    for m in SCORING:
        result[m] = scores[f'test_{m}'].mean()
        result[f'{m}_std'] = scores[f'test_{m}'].std()
    return result


def evaluate(model, X, y):
    """Métricas do modelo em um conjunto de dados (usado no teste)."""
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        'accuracy': accuracy_score(y, pred),
        'balanced_accuracy': balanced_accuracy_score(y, pred),
        'precision': precision_score(y, pred, zero_division=0),
        'recall': recall_score(y, pred),
        'f1': f1_score(y, pred, zero_division=0),
        'roc_auc': roc_auc_score(y, proba),
        'average_precision': average_precision_score(y, proba),
    }

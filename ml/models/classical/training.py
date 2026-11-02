"""Classifier construction, stratified model selection, and evaluation."""
import time
import warnings
from collections import defaultdict
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from ml.models.classical.folds import fold_features

EXPERIMENTS = {'bow_nb': ('bow', 'nb'), 'tfidf_nb': ('tfidf', 'nb'), 'bow_logistic': ('bow', 'logistic'), 'tfidf_logistic': ('tfidf', 'logistic'), 'tfidf_svm': ('tfidf', 'svm'), 'word2vec_logistic': ('word2vec', 'logistic')}
SEARCH = {'nb': (0.1, 1.0), 'logistic': (0.1, 1.0), 'svm': (0.1, 1.0)}


def make_model(kind, parameter):
    if parameter <= 0:
        raise ValueError('Hyperparameter must be positive')
    if kind == 'nb': return MultinomialNB(alpha=parameter)
    if kind == 'logistic': return LogisticRegression(C=parameter, solver='saga', max_iter=300, random_state=42)
    if kind == 'svm': return LinearSVC(C=parameter, random_state=42, dual='auto', max_iter=3000)
    raise ValueError(f'Unknown model: {kind}')


def fit_model(model, x, y):
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(x, y)
    return time.perf_counter()-start, [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]


def cross_validate(rows, labels, config, ngrams, w2v, folds=3, searches=SEARCH):
    labels = np.asarray(labels)
    if len(rows) != len(labels): raise ValueError('Feature and label counts differ')
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    results = defaultdict(lambda: defaultdict(list))
    for train_idx, val_idx in splitter.split(np.zeros(len(labels)), labels):
        features = fold_features([rows[i] for i in train_idx], [rows[i] for i in val_idx], config, ngrams, w2v)
        for name, (representation, kind) in EXPERIMENTS.items():
            x_train, x_val, _ = features[representation]
            for param in searches[kind]:
                model = make_model(kind, param)
                duration, notices = fit_model(model, x_train, labels[train_idx])
                results[name][param].append({'macro_f1': float(f1_score(labels[val_idx], model.predict(x_val), average='macro', zero_division=0)), 'fit_seconds': duration, 'convergence_warnings': notices})
    summary = {}
    for name, choices in results.items():
        scores = {str(param): {'folds': values, 'mean': float(np.mean([v['macro_f1'] for v in values])), 'std': float(np.std([v['macro_f1'] for v in values]))} for param, values in choices.items()}
        best = max(choices, key=lambda param: np.mean([v['macro_f1'] for v in choices[param]]))
        summary[name] = {'best_parameter': best, 'scores': scores}
    return summary


def evaluate(model, x, y, label_names):
    start = time.perf_counter()
    predicted = model.predict(x)
    elapsed = time.perf_counter()-start
    indices = list(range(len(label_names)))
    return {'accuracy': float(accuracy_score(y, predicted)), 'macro_f1': float(f1_score(y, predicted, labels=indices, average='macro', zero_division=0)), 'weighted_f1': float(f1_score(y, predicted, labels=indices, average='weighted', zero_division=0)), 'report': classification_report(y, predicted, labels=indices, target_names=label_names, output_dict=True, zero_division=0), 'confusion_matrix': confusion_matrix(y, predicted, labels=indices).tolist(), 'prediction_seconds': elapsed, 'seconds_per_document': elapsed/len(y) if len(y) else None}, predicted

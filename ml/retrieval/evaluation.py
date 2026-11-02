"""Evaluate stored rankings against explicitly supplied relevance judgments."""
import json
import statistics
from pathlib import Path
from ml.retrieval.metrics import evaluate


def load_judgments(path):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if not data.get('assessor') or not isinstance(data.get('judgments'),dict): raise ValueError('Missing assessor or judgments')
    for q,grades in data['judgments'].items():
        if not isinstance(q,str) or not isinstance(grades,dict): raise ValueError('Invalid judgments')
        if any(type(v) is not int or v not in (0,1,2) for v in grades.values()): raise ValueError('Invalid grade')
    return data


def compare(pool, judgments, k=5):
    per_query={}
    for query, methods in pool.items():
        if query not in judgments: raise ValueError(f'Missing judgments for {query}')
        per_query[query]={method:evaluate([row['arxiv_id'] for row in rows], judgments[query], k) for method,rows in methods.items()}
    if not per_query: raise ValueError('No queries')
    methods=set(next(iter(per_query.values())))
    if any(set(results)!=methods for results in per_query.values()): raise ValueError('Methods differ between queries')
    averages={method:{metric:statistics.mean(per_query[q][method][metric] for q in per_query) for metric in ('precision','recall','mrr','ndcg')} for method in methods}
    return per_query,averages

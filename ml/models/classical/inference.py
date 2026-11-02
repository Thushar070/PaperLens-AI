"""Inference using saved Phase 2 representations and Phase 3 models."""
import json
from pathlib import Path
import joblib
import numpy as np
from gensim.models import Word2Vec
from ml.models.classical.training import EXPERIMENTS
from ml.preprocessing.text import document, tokens
from ml.representations.features import document_vector


def predict(title, abstract, experiment, data_dir='artifacts/data'):
    if experiment not in EXPERIMENTS: raise ValueError('Unknown experiment')
    text=document({'title':title,'abstract':abstract})
    if not text.strip(): raise ValueError('Title and abstract cannot both be empty')
    root=Path(data_dir)
    model=joblib.load(root/'phase3'/'models'/f'{experiment}.joblib')
    rep=EXPERIMENTS[experiment][0]
    if rep=='word2vec':
        w2v=Word2Vec.load(str(root/'phase2'/'word2vec.model'))
        x=np.asarray([document_vector(w2v,tokens(text))])
    else:
        x=joblib.load(root/'phase2'/f'{rep}.joblib').transform([text])
    label=int(model.predict(x)[0])
    labels=json.loads((root/'phase1'/'metadata.json').read_text(encoding='utf-8'))['label_mapping']
    result={'label':label,'category':labels[str(label)]}
    if hasattr(model,'predict_proba'):
        result['probabilities']={labels[str(int(k))]:float(v) for k,v in zip(model.classes_,model.predict_proba(x)[0])}
    elif hasattr(model,'decision_function'):
        values=np.atleast_1d(model.decision_function(x)[0])
        result['decision_scores']={labels[str(int(k))]:float(v) for k,v in zip(model.classes_,values)}
    return result

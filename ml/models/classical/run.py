"""Resumable Phase 3 experiments using immutable Phase 1 and 2 artifacts."""
import hashlib
import json
import os
from pathlib import Path
import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from ml.representations.phase2 import read_manifest
from ml.models.classical.folds import fold_features
from ml.models.classical.training import EXPERIMENTS, SEARCH, make_model, fit_model, evaluate

ROOT = Path('artifacts/data')
OUT = ROOT/'phase3'
CHECKPOINT = Path('docs/phase3_execution_checkpoint.json')

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(tmp, path)

def update(task, **changes):
    data = json.loads(CHECKPOINT.read_text(encoding='utf-8'))
    data['current_task'] = task
    data.update(changes)
    save(CHECKPOINT, data)

def verify_inputs():
    data = json.loads(CHECKPOINT.read_text(encoding='utf-8'))
    for filename, expected in {**data['input_sha256'], **data.get('output_sha256', {})}.items():
        actual = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
        if actual != expected: raise RuntimeError(f'Input changed: {filename}')

def params(meta):
    return {k:meta['word2vec_config'][k] for k in ('vector_size','window','min_count','epochs','seed')}

def load_rows(split, metadata):
    rows = read_manifest(ROOT/'phase1'/f'{split}.jsonl')
    if len(rows) != metadata['split_sizes'][split]: raise ValueError('Split length mismatch')
    return rows

def cv(train, metadata):
    y = np.array([r['label'] for r in train])
    config = metadata['classical_config']
    for fold, (tr, va) in enumerate(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(np.zeros(len(y)),y)):
        pending = [(name,p) for name,(_,kind) in EXPERIMENTS.items() for p in SEARCH[kind] if not (OUT/'cv'/f'{fold}_{name}_{p}.json').exists()]
        if not pending: continue
        update(f'Cross-validation fold {fold+1}: preparing fold-only features')
        features = fold_features([train[i] for i in tr], [train[i] for i in va], config, metadata['ngram_range'], params(metadata))
        for name, p in pending:
            representation, kind = EXPERIMENTS[name]
            x, v, _ = features[representation]
            model = make_model(kind,p)
            update(f'Cross-validation fold {fold+1}: {name} parameter {p}')
            duration, notices = fit_model(model,x,y[tr])
            score = f1_score(y[va],model.predict(v),average='macro',zero_division=0)
            save(OUT/'cv'/f'{fold}_{name}_{p}.json', {'fold':fold,'experiment':name,'parameter':p,'macro_f1':float(score),'fit_seconds':duration,'warnings':notices})
    summary = {}
    for name,(_,kind) in EXPERIMENTS.items():
        choices={}
        for p in SEARCH[kind]:
            values=[json.loads((OUT/'cv'/f'{fold}_{name}_{p}.json').read_text(encoding='utf-8')) for fold in range(3)]
            scores=[item['macro_f1'] for item in values]
            choices[str(p)]={'mean':float(np.mean(scores)),'std':float(np.std(scores)),'folds':values}
        eligible=[p for p in SEARCH[kind] if all(not item['warnings'] for item in choices[str(p)]['folds'])]
        if not eligible: raise RuntimeError(f'No converged candidates for {name}; increase iterations and rerun affected candidates')
        best=max(eligible,key=lambda p:choices[str(p)]['mean'])
        summary[name]={'best_parameter':best,'scores':choices}
    save(OUT/'cross_validation.json',summary)
    return summary

def matrix(split, representation):
    path=ROOT/'phase2'/f'{split}_{representation}'
    return np.load(str(path)+'.npy') if representation=='word2vec' else load_npz(str(path)+'.npz')

def train_final(train, validation, names, results):
    y=np.array([r['label'] for r in train]); vy=np.array([r['label'] for r in validation])
    for name,(rep,kind) in EXPERIMENTS.items():
        model_path=OUT/'models'/f'{name}.joblib'; metrics_path=OUT/'validation'/f'{name}.json'
        if model_path.exists() and metrics_path.exists():
            joblib.load(model_path); continue
        update(f'Full training and validation: {name}')
        x=matrix('train',rep); v=matrix('validation',rep)
        if x.shape[0]!=len(y) or v.shape[0]!=len(vy): raise ValueError('Feature/label alignment mismatch')
        model=make_model(kind,results[name]['best_parameter']); duration, notices=fit_model(model,x,y)
        if notices: raise RuntimeError(f'Final model {name} did not converge: {notices}')
        model_path.parent.mkdir(parents=True,exist_ok=True)
        joblib.dump(model, model_path)
        metrics,pred=evaluate(model,v,vy,names)
        metrics.update({'training_seconds':duration,'warnings':notices,'model_bytes':model_path.stat().st_size})
        save(metrics_path,metrics)
        save(OUT/'validation'/f'{name}_predictions.json',[{'arxiv_id':r['arxiv_id'],'title':r['title'],'actual':int(a),'predicted':int(b)} for r,a,b in zip(validation,vy,pred)])

def final_test(rows,names):
    y=np.array([r['label'] for r in rows])
    for name,(rep,_) in EXPERIMENTS.items():
        target=OUT/'test'/f'{name}.json'
        if target.exists(): continue
        update(f'Final test: {name}')
        model=joblib.load(OUT/'models'/f'{name}.joblib')
        x=matrix('test',rep)
        if x.shape[0]!=len(y): raise ValueError('Test alignment mismatch')
        metrics,pred=evaluate(model,x,y,names)
        save(target,metrics)
        save(OUT/'test'/f'{name}_predictions.json',[{'arxiv_id':r['arxiv_id'],'actual':int(a),'predicted':int(b)} for r,a,b in zip(rows,y,pred)])

def main():
    verify_inputs()
    meta1=json.loads((ROOT/'phase1'/'metadata.json').read_text(encoding='utf-8'))
    meta2=json.loads((ROOT/'phase2'/'metadata.json').read_text(encoding='utf-8'))
    names=[meta1['label_mapping'][str(i)] for i in range(len(meta1['label_mapping']))]
    train=load_rows('train',meta1)
    results=cv(train,meta2)
    val=load_rows('validation',meta1)
    train_final(train,val,names,results)
    validation_scores = {name: json.loads((OUT/'validation'/f'{name}.json').read_text(encoding='utf-8'))['macro_f1'] for name in EXPERIMENTS}
    winner = max(validation_scores, key=validation_scores.get)
    selection_path = OUT/'model_selection.json'
    if selection_path.exists():
        existing = json.loads(selection_path.read_text(encoding='utf-8'))
        if existing != {'selected': winner, 'validation_macro_f1': validation_scores}: raise ValueError('Selection does not match saved validation results')
    else:
        save(selection_path, {'selected': winner, 'validation_macro_f1': validation_scores})
    update('All model configurations finalized; final test evaluation')
    test=load_rows('test',meta1)
    final_test(test,names)
    update('Training and evaluation complete; remaining verification and reporting')

if __name__=='__main__': main()

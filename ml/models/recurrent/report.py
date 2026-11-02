"""Produce Phase 5 comparisons, confusion plots and error analysis."""
import json
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ml.models.recurrent.train import ROOT,OUT,ARCHES,NAMES

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    selections=read(OUT/'selection.json')
    val={n:read(OUT/'validation'/f'{n}.json') for n in ARCHES};test={n:read(OUT/'test'/f'{n}.json') for n in ARCHES}
    vbase=read(ROOT/'phase3/validation/tfidf_svm.json');tbase=read(ROOT/'phase3/test/tfidf_svm.json')
    lines=['# Phase 5 measured comparison','','Validation selects each checkpoint and learning rate; test results are reported only after selection. Classifier timings for the Phase 3 model exclude preprocessing, whereas recurrent timings include batch handling and transfers.','', '| Model | Validation macro F1 | Test macro F1 | Test accuracy | Test weighted F1 | Training seconds | Checkpoint bytes |','|---|---:|---:|---:|---:|---:|---:|',f"| TF-IDF + Linear SVM | {vbase['macro_f1']:.4f} | {tbase['macro_f1']:.4f} | {tbase['accuracy']:.4f} | {tbase['weighted_f1']:.4f} | {vbase['training_seconds']:.2f} | {vbase['model_bytes']} |"]
    for n in ARCHES:
        v=val[n];t=test[n];lines.append(f"| {n} | {v['macro_f1']:.4f} | {t['macro_f1']:.4f} | {t['accuracy']:.4f} | {t['weighted_f1']:.4f} | {v['training_seconds']:.2f} | {v['checkpoint_bytes']} |")
    Path('docs/phase5_comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    errors={n:[r for r in read(OUT/'test'/f'{n}_predictions.json') if r['actual']!=r['predicted']] for n in ARCHES}
    shared=Counter(r['arxiv_id'] for e in errors.values() for r in e)
    analysis={'confusions':{n:dict(Counter(f"{NAMES[r['actual']]} -> {NAMES[r['predicted']]}" for r in errors[n])) for n in ARCHES},'shared_misclassifications':[p for p,count in shared.items() if count==len(ARCHES)],'errors':errors}
    (OUT/'error_analysis.json').write_text(json.dumps(analysis,indent=2,ensure_ascii=False),encoding='utf-8')
    fig,axes=plt.subplots(2,3,figsize=(14,9))
    for ax,n in zip(axes.flat,ARCHES):
        mat=test[n]['confusion_matrix'];ax.imshow(mat,cmap='Blues');ax.set_title(n);ax.set_xticks(range(4),NAMES,rotation=35,ha='right');ax.set_yticks(range(4),NAMES)
        for i in range(4):
            for j in range(4): ax.text(j,i,str(mat[i][j]),ha='center',va='center')
    axes.flat[-1].axis('off');fig.tight_layout();fig.savefig(OUT/'test_confusion.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4));ax.bar(ARCHES,[test[n]['macro_f1'] for n in ARCHES]);ax.axhline(tbase['macro_f1'],color='red',label='TF-IDF + SVM');ax.set_ylim(0,1);ax.set_ylabel('Test macro F1');ax.tick_params(axis='x',rotation=25);ax.legend();fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
    print('Shared misclassifications:',len(analysis['shared_misclassifications']))

if __name__=='__main__':main()

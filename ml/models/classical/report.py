"""Generate Phase 3 comparison and error analysis from completed results."""
import json
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ml.models.classical.training import EXPERIMENTS

ROOT=Path('artifacts/data/phase3')

def read(path): return json.loads(path.read_text(encoding='utf-8'))

def main():
    cv=read(ROOT/'cross_validation.json')
    names=list(EXPERIMENTS)
    validation={n:read(ROOT/'validation'/f'{n}.json') for n in names}
    test={n:read(ROOT/'test'/f'{n}.json') for n in names}
    labels=list(read(Path('artifacts/data/phase1/metadata.json'))['label_mapping'].values())
    fig,axes=plt.subplots(2,2,figsize=(15,10))
    axes[0,0].bar(names,[validation[n]['macro_f1'] for n in names]); axes[0,0].set_title('Validation macro F1')
    axes[0,1].bar(names,[cv[n]['scores'][str(cv[n]['best_parameter'])]['mean'] for n in names]); axes[0,1].set_title('Training cross-validation macro F1')
    width=.12
    for i,label in enumerate(labels): axes[1,0].bar([j+i*width for j in range(len(names))],[validation[n]['report'][label]['f1-score'] for n in names],width,label=label)
    axes[1,0].set_xticks([j+1.5*width for j in range(len(names))],names); axes[1,0].legend(); axes[1,0].set_title('Validation per-class F1')
    axes[1,1].axis('off')
    for ax in axes.flatten()[:3]: ax.tick_params(axis='x',rotation=30); ax.set_ylim(0,1)
    fig.tight_layout(); fig.savefig(ROOT/'comparison.png',dpi=150); plt.close(fig)
    for split,metrics in [('validation',validation),('test',test)]:
        fig,axes=plt.subplots(2,3,figsize=(14,9))
        for ax,name in zip(axes.flat,names):
            mat=metrics[name]['confusion_matrix']; ax.imshow(mat,cmap='Blues'); ax.set_title(name)
            ax.set_xticks(range(4),labels,rotation=40,ha='right'); ax.set_yticks(range(4),labels)
            for i in range(4):
                for j in range(4): ax.text(j,i,str(mat[i][j]),ha='center',va='center')
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        fig.tight_layout(); fig.savefig(ROOT/f'{split}_confusion.png',dpi=150); plt.close(fig)
    errors={}
    for name in names:
        predictions=read(ROOT/'validation'/f'{name}_predictions.json')
        errors[name]=[r for r in predictions if r['actual']!=r['predicted']]
    common=Counter(r['arxiv_id'] for records in errors.values() for r in records)
    analysis={'errors':errors,'most_common_confusions':{name:Counter(f"{labels[r['actual']]} -> {labels[r['predicted']]}" for r in rows) for name,rows in errors.items()},'papers_misclassified_by_all_models':[k for k,v in common.items() if v==len(names)]}
    (ROOT/'error_analysis.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')

if __name__=='__main__': main()

"""Reload recurrent models and apply the saved Phase 2 vocabulary."""
import json
from pathlib import Path
import torch
from ml.models.recurrent.model import RecurrentClassifier
from ml.preprocessing.text import document,tokens
from ml.preprocessing.vocabulary import encode


def predict(title,abstract,architecture,root='artifacts/data',device=None):
    root=Path(root);text=document({'title':title,'abstract':abstract})
    if not text.strip(): raise ValueError('Title and abstract cannot both be empty')
    selection=json.loads((root/'phase5/selection.json').read_text(encoding='utf-8'))
    if architecture not in selection: raise ValueError('Unknown model')
    winner=selection[architecture];lr=winner['config']['learning_rate']
    path=root/'phase5'/'runs'/f'{architecture}_{lr}'/'best.pt'
    device=torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    checkpoint=torch.load(path,map_location=device,weights_only=False); cfg=checkpoint['config']
    model=RecurrentClassifier(**{k:cfg[k] for k in ('architecture','vocab_size','embedding_dim','hidden_dim','classes','dropout')}).to(device);model.load_state_dict(checkpoint['model']);model.eval()
    vocab=json.loads((root/'phase2/vocabulary.json').read_text(encoding='utf-8'))
    length=json.loads((root/'phase2/metadata.json').read_text(encoding='utf-8'))['sequence_length']
    x=torch.tensor([encode(tokens(text),vocab,length)],device=device)
    with torch.no_grad(): probabilities=torch.softmax(model(x),dim=1)[0].cpu().tolist()
    labels=json.loads((root/'phase1/metadata.json').read_text(encoding='utf-8'))['label_mapping'];index=max(range(len(probabilities)),key=probabilities.__getitem__)
    return {'label':index,'category':labels[str(index)],'probabilities':{labels[str(i)]:p for i,p in enumerate(probabilities)},'device':str(device)}

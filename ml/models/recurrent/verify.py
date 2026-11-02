"""Verify selected model reloads and measure repeated inference timings."""
import json,time,hashlib
from pathlib import Path
import torch
from ml.models.recurrent.model import RecurrentClassifier
from ml.models.recurrent.train import ROOT,OUT,CHECK,ARCHES,load_split,atomic_json
from ml.models.recurrent.inference import predict
from ml.preprocessing.text import document,tokens
from ml.preprocessing.vocabulary import encode

def main():
    state=json.loads(CHECK.read_text(encoding='utf-8'))
    for f,h in state['input_sha256'].items():
        if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h: raise RuntimeError(f'Input changed: {f}')
    selection=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    vocab=json.loads((ROOT/'phase2/vocabulary.json').read_text(encoding='utf-8')); x,y,rows=load_split('validation')
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');results={}
    for arch in ARCHES:
        lr=selection[arch]['config']['learning_rate'];path=OUT/'runs'/f'{arch}_{lr}'/'best.pt'
        ckpt=torch.load(path,map_location=device,weights_only=False);cfg=ckpt['config'];model=RecurrentClassifier(**{k:cfg[k] for k in ('architecture','vocab_size','embedding_dim','hidden_dim','classes','dropout')}).to(device);model.load_state_dict(ckpt['model']);model.eval()
        sample=x[:64].to(device);times=[]
        with torch.no_grad():
            model(sample)
            for i in range(5):
                if device.type=='cuda':torch.cuda.synchronize()
                start=time.perf_counter();model(sample)
                if device.type=='cuda':torch.cuda.synchronize()
                times.append(time.perf_counter()-start)
        raw=rows[0]; elapsed=[]
        for _ in range(5):
            t0=time.perf_counter(); token_ids=encode(tokens(document(raw)),vocab,290); bx=torch.tensor([token_ids],device=device)
            with torch.no_grad(): pred=int(model(bx).argmax(1).item())
            elapsed.append(time.perf_counter()-t0)
        saved=predict(raw['title'],raw['abstract'],arch,device=str(device)); assert pred==saved['label']
        results[arch]={'device':str(device),'model_only_batch_size':64,'model_only_seconds':times,'end_to_end_single_seconds_repeated':elapsed,'inference':saved,'note':'single text timing includes tokenization, encoding, transfer and forward; excludes model loading'}
    atomic_json(OUT/'verification.json',{'input_hashes_checked':len(state['input_sha256']),'models':results})
    print('Verified',len(results),'models and',len(state['input_sha256']),'input hashes')

if __name__=='__main__':main()

"""Train and evaluate the Phase 5 recurrent models on saved sequences."""
import json, os, random, time, hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from ml.models.recurrent.model import RecurrentClassifier

ROOT=Path('artifacts/data'); OUT=ROOT/'phase5'; CHECK=Path('docs/phase5_execution_checkpoint.json')
NAMES=['Algorithms','NLP/AI','Systems/Infra','Vision']
ARCHES=['rnn','lstm','gru','bilstm','bilstm_attention']

def atomic_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2),encoding='utf-8');os.replace(temp,path)

def atomic_torch(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');torch.save(data,temp);os.replace(temp,path)

def update(task, **changes):
    d=json.loads(CHECK.read_text(encoding='utf-8'));d['task']=task;d.update(changes);atomic_json(CHECK,d)

def load_split(name):
    x=np.load(ROOT/'phase2'/f'{name}_sequences.npy')
    with (ROOT/'phase1'/f'{name}.jsonl').open(encoding='utf-8') as f: rows=[json.loads(line) for line in f]
    if x.shape!=(len(rows),290): raise ValueError('Invalid sequence alignment')
    return torch.from_numpy(x.astype(np.int64)),torch.tensor([r['label'] for r in rows]),rows

def batches(x,y,batch,shuffle=False,seed=42):
    return DataLoader(TensorDataset(x,y),batch_size=batch,shuffle=shuffle,generator=torch.Generator().manual_seed(seed),num_workers=0)

def run_epoch(model,loader,device,optimizer=None):
    model.train(optimizer is not None); total=0; losses=0; pred=[]; truth=[]
    with torch.set_grad_enabled(optimizer is not None):
        for x,y in loader:
            length=max(1,int(x.ne(0).sum(1).max()))
            x=x[:,:length].to(device);y=y.to(device)
            if next(model.parameters()).device!=device or x.device!=device: raise RuntimeError('Device mismatch')
            if optimizer: optimizer.zero_grad(set_to_none=True)
            logits=model(x);loss=nn.functional.cross_entropy(logits,y)
            if optimizer:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            losses+=loss.item()*len(y);total+=len(y);pred.extend(logits.argmax(1).cpu().tolist());truth.extend(y.cpu().tolist())
    return {'loss':losses/total,'macro_f1':float(f1_score(truth,pred,average='macro',zero_division=0))}

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def fit(arch,lr,train,val,device):
    name=f'{arch}_{lr}'; folder=OUT/'runs'/name; folder.mkdir(parents=True,exist_ok=True)
    latest=folder/'latest.pt'; best=folder/'best.pt'; done=folder/'completed.json'
    if done.exists() and latest.exists() and best.exists(): return json.loads(done.read_text())
    cfg={'architecture':arch,'vocab_size':32887,'embedding_dim':64,'hidden_dim':64,'classes':4,'dropout':.2,'learning_rate':lr,'batch_size':64,'seed':42,'epochs':4,'patience':2}
    seed_all(42); model=RecurrentClassifier(**{k:cfg[k] for k in ('architecture','vocab_size','embedding_dim','hidden_dim','classes','dropout')}).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr); history=[];start_epoch=0; best_f1=-1.; bad=0;elapsed=0.
    if latest.exists():
        state=torch.load(latest,map_location=device,weights_only=False)
        if state['config']!=cfg: raise ValueError('Run configuration differs from saved checkpoint')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);history=state['history'];start_epoch=state['epoch']+1;best_f1=state['best_f1'];bad=state['bad'];elapsed=state['elapsed'];torch.set_rng_state(state['rng'].cpu())
        if device.type=='cuda': torch.cuda.set_rng_state(state['cuda_rng'].cpu())
    start=time.perf_counter()
    for epoch in range(start_epoch,cfg['epochs']):
        update(f'{name} epoch {epoch+1}')
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        tr=run_epoch(model,batches(*train,64,True,42+epoch),device,optimizer); va=run_epoch(model,batches(*val,64),device)
        history.append({'epoch':epoch+1,'train':tr,'validation':va,'device':str(device),'gpu_peak_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None})
        if va['macro_f1']>best_f1:
            best_f1=va['macro_f1'];bad=0;atomic_torch(best,{'config':cfg,'model':model.state_dict(),'validation':va,'epoch':epoch+1})
        else: bad+=1
        state={'config':cfg,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'history':history,'epoch':epoch,'best_f1':best_f1,'bad':bad,'elapsed':elapsed+time.perf_counter()-start,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state() if device.type=='cuda' else None}
        atomic_torch(latest,state);atomic_json(folder/'history.json',history)
        print(name,epoch+1,'train',tr,'validation',va,flush=True)
        if bad>=cfg['patience']: break
    result={'config':cfg,'best_validation_macro_f1':best_f1,'training_seconds':elapsed+time.perf_counter()-start,'device':str(device),'epochs_completed':len(history)}
    atomic_json(done,result);return result

def evaluate(model,x,y,rows,device):
    model.eval(); outputs=[]; timings=[]
    for repeat in range(3):
        pred=[];start=time.perf_counter()
        with torch.no_grad():
            for bx,by in batches(x,y,64):
                length=max(1,int(bx.ne(0).sum(1).max())); bx=bx[:,:length].to(device);pred.extend(model(bx).argmax(1).cpu().tolist())
        if device.type=='cuda': torch.cuda.synchronize()
        timings.append(time.perf_counter()-start);outputs=pred
    truth=y.tolist(); report=classification_report(truth,outputs,labels=list(range(4)),target_names=NAMES,output_dict=True,zero_division=0)
    return {'accuracy':float(accuracy_score(truth,outputs)),'macro_f1':report['macro avg']['f1-score'],'weighted_f1':report['weighted avg']['f1-score'],'report':report,'confusion_matrix':confusion_matrix(truth,outputs,labels=list(range(4))).tolist(),'batch_size':64,'inference_seconds_repeated':timings,'latency_description':'Includes batch slicing, device transfer, model and CPU prediction transfer; excludes text tokenization'}, [{'arxiv_id':r['arxiv_id'],'title':r['title'],'actual':int(a),'predicted':int(b)} for r,a,b in zip(rows,truth,outputs)]

def main():
    state=json.loads(CHECK.read_text(encoding='utf-8'))
    for f,h in state['input_sha256'].items():
        if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h: raise RuntimeError(f'Input changed: {f}')
    meta=json.loads((ROOT/'phase2'/'metadata.json').read_text(encoding='utf-8')); vocab=json.loads((ROOT/'phase2'/'vocabulary.json').read_text(encoding='utf-8'))
    if len(vocab)!=32887 or meta['sequence_length']!=290 or vocab['<PAD>']!=0 or vocab['<UNK>']!=1: raise ValueError('Unexpected Phase 2 vocabulary')
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    update('Loading training and validation',device=str(device),torch=torch.__version__,cuda=torch.version.cuda)
    tx,ty,_=load_split('train');vx,vy,vr=load_split('validation')
    selections={}
    for arch in ARCHES:
        runs=[fit(arch,lr,(tx,ty),(vx,vy),device) for lr in (0.001,0.002)]
        winner=max(runs,key=lambda x:x['best_validation_macro_f1']); selections[arch]=winner
        path=OUT/'runs'/f"{arch}_{winner['config']['learning_rate']}"/'best.pt'
        checkpoint=torch.load(path,map_location=device,weights_only=False);cfg=checkpoint['config'];model=RecurrentClassifier(**{k:cfg[k] for k in ('architecture','vocab_size','embedding_dim','hidden_dim','classes','dropout')}).to(device);model.load_state_dict(checkpoint['model'])
        metrics,preds=evaluate(model,vx,vy,vr,device);metrics.update({'device':str(device),'parameters':sum(p.numel() for p in model.parameters()),'checkpoint_bytes':path.stat().st_size,'training_seconds':winner['training_seconds'],'checkpoint':str(path)})
        atomic_json(OUT/'validation'/f'{arch}.json',metrics);atomic_json(OUT/'validation'/f'{arch}_predictions.json',preds)
    atomic_json(OUT/'selection.json',selections)
    update('All selections finalized; evaluating test split')
    sx,sy,sr=load_split('test')
    for arch,winner in selections.items():
        result=OUT/'test'/f'{arch}.json'
        if result.exists() and (OUT/'test'/f'{arch}_predictions.json').exists(): continue
        path=OUT/'runs'/f"{arch}_{winner['config']['learning_rate']}"/'best.pt';checkpoint=torch.load(path,map_location=device,weights_only=False);cfg=checkpoint['config'];model=RecurrentClassifier(**{k:cfg[k] for k in ('architecture','vocab_size','embedding_dim','hidden_dim','classes','dropout')}).to(device);model.load_state_dict(checkpoint['model'])
        metrics,preds=evaluate(model,sx,sy,sr,device);atomic_json(result,metrics);atomic_json(OUT/'test'/f'{arch}_predictions.json',preds)
    update('Training and test evaluation complete')

if __name__=='__main__': main()

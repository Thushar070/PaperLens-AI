"""Read-only local retrieval using Phase 2 representations."""
from pathlib import Path
import joblib
import numpy as np
from scipy.sparse import load_npz, vstack
from sklearn.preprocessing import normalize
from gensim.models import Word2Vec
from ml.preprocessing.text import tokens
from ml.representations.features import document_vector
from ml.representations.phase2 import read_manifest

class Retriever:
    def __init__(self, root='artifacts/data'):
        root=Path(root)
        self.records=[]
        matrices=[]; vectors=[]
        for split in ('train','validation','test'):
            rows=read_manifest(root/'phase1'/f'{split}.jsonl')
            mat=load_npz(root/'phase2'/f'{split}_tfidf.npz').tocsr()
            vec=np.load(root/'phase2'/f'{split}_word2vec.npy')
            if len(rows)!=mat.shape[0] or len(rows)!=len(vec): raise ValueError('Record and vector counts differ')
            self.records.extend(rows); matrices.append(mat); vectors.append(vec)
        ids=[r['arxiv_id'] for r in self.records]
        if len(ids)!=len(set(ids)): raise ValueError('Duplicate paper identifiers')
        self.tfidf=normalize(vstack(matrices,format='csr'),copy=False)
        self.word2vec=normalize(np.concatenate(vectors),copy=False)
        self.vectorizer=joblib.load(root/'phase2'/'tfidf.joblib')
        self.model=Word2Vec.load(str(root/'phase2'/'word2vec.model'))
        if self.tfidf.shape[1]!=len(self.vectorizer.vocabulary_) or self.word2vec.shape[1]!=self.model.vector_size: raise ValueError('Representation dimensions differ')

    def search(self, query, method='tfidf', k=10):
        if not isinstance(query,str): raise TypeError('Query must be a string')
        if not isinstance(k,int) or isinstance(k,bool) or k<0: raise ValueError('k must be a nonnegative integer')
        if method not in ('tfidf','word2vec'): raise ValueError('Unknown retrieval method')
        if not query.strip() or k==0: return []
        if method=='tfidf':
            q=self.vectorizer.transform([query])
            if q.nnz==0: return []
            scores=np.asarray((self.tfidf @ normalize(q).T).toarray()).ravel()
        else:
            q=document_vector(self.model,tokens(query))
            norm=np.linalg.norm(q)
            if not norm: return []
            scores=self.word2vec @ (q/norm)
        if method=='tfidf':
            scores[np.asarray(self.tfidf.getnnz(axis=1)).ravel()==0] = np.nan
        else:
            scores[np.linalg.norm(self.word2vec,axis=1)==0] = np.nan
        return rank(self.records,scores,k)


def rank(records, scores, k):
    scores=np.asarray(scores).reshape(-1)
    if len(records)!=len(scores): raise ValueError('Scores and records differ in length')
    valid=[i for i,s in enumerate(scores) if np.isfinite(s)]
    valid.sort(key=lambda i:(-scores[i],records[i]['arxiv_id']))
    return [{'arxiv_id':records[i]['arxiv_id'],'title':records[i]['title'],'abstract':records[i]['abstract'],'label':records[i]['label'],'score':float(scores[i])} for i in valid[:k]]

import unittest
import numpy as np
from ml.retrieval.engine import rank, Retriever
from ml.retrieval.metrics import evaluate

class RetrievalTests(unittest.TestCase):
    def test_ranking(self):
        rows=[{'arxiv_id':'b','title':'b','abstract':'','label':0},{'arxiv_id':'a','title':'a','abstract':'','label':0},{'arxiv_id':'c','title':'c','abstract':'','label':0}]
        self.assertEqual([r['arxiv_id'] for r in rank(rows,[.5,.5,np.nan],3)],['a','b'])
        self.assertEqual(rank(rows,[0,-1,np.nan],0),[])
        self.assertEqual([r['arxiv_id'] for r in rank(rows,[0,-1,np.nan],3)],['b','a'])
    def test_metrics(self):
        m=evaluate(['x','a','b'],{'a':2,'b':1,'x':0},2)
        self.assertEqual(m['precision'],.5); self.assertEqual(m['recall'],.5); self.assertEqual(m['mrr'],.5)
        self.assertAlmostEqual(m['ndcg'],(3/np.log2(3))/(3+1/np.log2(3)))
        with self.assertRaises(ValueError): evaluate(['a'],{'a':0},1)
        with self.assertRaises(ValueError): evaluate(['a','a'],{'a':1},2)
    def test_real_artifacts(self):
        r=Retriever(); self.assertEqual(len(r.records),19999)
        for method in ('tfidf','word2vec'):
            got=r.search('graph neural networks',method,5)
            self.assertEqual(len(got),5); self.assertTrue(all(x['arxiv_id'] for x in got))
            self.assertEqual(r.search('',method),[])
            self.assertEqual(r.search('zzzxqvnotinvocabulary',method),[])
            self.assertEqual(r.search('graph',method,0),[])
            self.assertEqual(got,r.search('graph neural networks',method,5))
        with self.assertRaises(ValueError): r.search('test',k=-1)

if __name__=='__main__': unittest.main()

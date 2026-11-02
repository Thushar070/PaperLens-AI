import json,tempfile,unittest
from pathlib import Path
from ml.retrieval.evaluation import load_judgments,compare

class JudgmentsTests(unittest.TestCase):
    def test_read_and_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'judgments.json'; f.write_text(json.dumps({'assessor':'fixture','judgments':{'q':{'a':2,'b':0}}}))
            d=load_judgments(f)
            per,avg=compare({'q':{'method':[{'arxiv_id':'a'},{'arxiv_id':'b'}]}},d['judgments'],2)
            self.assertEqual(avg['method']['precision'],.5); self.assertEqual(avg['method']['ndcg'],1)
            with self.assertRaises(ValueError): compare({'missing':{}},d['judgments'])
            f.write_text(json.dumps({'assessor':'fixture','judgments':{'q':{'a':3}}}))
            with self.assertRaises(ValueError): load_judgments(f)

if __name__=='__main__': unittest.main()

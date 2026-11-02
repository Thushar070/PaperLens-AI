import json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.sparse import save_npz
from ml.models.classical import run
from ml.representations.features import vectorizers, train_word2vec, document_vector
from ml.preprocessing.text import document,tokens

class PipelineTests(unittest.TestCase):
    def test_full_fixture_and_selection_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); output=root/'phase3'; checkpoint=root/'checkpoint.json'
            for part in ('phase1','phase2'): (root/part).mkdir()
            rows=[{'arxiv_id':str(i),'title':('apple' if i%2==0 else 'banana')+' topic','abstract':'common paper text','label':i%2} for i in range(24)]
            groups={'train':rows[:16], 'validation':rows[16:20], 'test':rows[20:]}
            mapping={'0':'apple','1':'banana'}
            m1={'label_mapping':mapping,'split_sizes':{k:len(v) for k,v in groups.items()}}
            w={'vector_size':4,'window':2,'min_count':1,'epochs':2,'seed':42}
            m2={'classical_config':{},'ngram_range':[1,1],'word2vec_config':w}
            (root/'phase1/metadata.json').write_text(json.dumps(m1),encoding='utf-8')
            (root/'phase2/metadata.json').write_text(json.dumps(m2),encoding='utf-8')
            checkpoint.write_text(json.dumps({'input_sha256':{}}),encoding='utf-8')
            bow,tfidf=vectorizers(ngram_range=(1,1)); docs=[document(r) for r in groups['train']]
            bow.fit(docs); tfidf.fit(docs)
            wmodel=train_word2vec([tokens(t) for t in docs],**w)
            for split,rs in groups.items():
                (root/'phase1'/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rs),encoding='utf-8')
                texts=[document(r) for r in rs]
                for name,vec in [('bow',bow),('tfidf',tfidf)]: save_npz(root/'phase2'/f'{split}_{name}.npz',vec.transform(texts))
                np.save(root/'phase2'/f'{split}_word2vec.npy',np.array([document_vector(wmodel,tokens(t)) for t in texts]))
            original=run.final_test
            def verify_selection(*args):
                self.assertTrue((output/'model_selection.json').exists())
                return original(*args)
            with patch.object(run,'ROOT',root),patch.object(run,'OUT',output),patch.object(run,'CHECKPOINT',checkpoint),patch.object(run,'final_test',side_effect=verify_selection): run.main()
            self.assertEqual(len(list((output/'models').glob('*.joblib'))),6)
            self.assertEqual(len(list((output/'test').glob('*_predictions.json'))),6)
            self.assertEqual(len(json.loads((output/'cross_validation.json').read_text(encoding='utf-8'))),6)

if __name__=='__main__': unittest.main()

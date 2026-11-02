import json
import tempfile
import unittest
from pathlib import Path
import joblib
from ml.models.classical.inference import predict
from ml.models.classical.training import make_model
from ml.representations.features import vectorizers

class InferenceTests(unittest.TestCase):
    def test_saved_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('phase1','phase2','phase3/models'): (root/name).mkdir(parents=True,exist_ok=True)
            (root/'phase1/metadata.json').write_text(json.dumps({'label_mapping':{'0':'alpha','1':'beta'}}),encoding='utf-8')
            vec,_=vectorizers(ngram_range=(1,1))
            x=vec.fit_transform(['alpha apple','beta banana'])
            model=make_model('nb',1).fit(x,[0,1])
            joblib.dump(vec,root/'phase2/bow.joblib')
            joblib.dump(model,root/'phase3/models/bow_nb.joblib')
            result=predict('alpha','apple unseen','bow_nb',root)
            self.assertEqual(result['category'],'alpha')
            self.assertAlmostEqual(sum(result['probabilities'].values()),1)

if __name__=='__main__': unittest.main()

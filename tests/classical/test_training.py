import unittest
from ml.models.classical.training import make_model, cross_validate

class TrainingTests(unittest.TestCase):
    def test_models_and_validation(self):
        for name in ('nb','logistic','svm'):
            self.assertIsNotNone(make_model(name, 1))
        with self.assertRaises(ValueError): make_model('unknown', 1)
        with self.assertRaises(ValueError): make_model('nb', 0)

    def test_all_models_fold_training(self):
        rows = [{'title': ('alpha' if i % 2 == 0 else 'beta') + ' common', 'abstract':'paper topic'} for i in range(12)]
        labels = [i % 2 for i in range(12)]
        config = {'vector_size':4,'window':2,'min_count':1,'epochs':2,'seed':42}
        result = cross_validate(rows, labels, {}, (1,1), config, folds=2, searches={'nb':(1,), 'logistic':(1,), 'svm':(1,)})
        self.assertEqual(len(result), 6)
        self.assertTrue(all(len(r['scores']['1']['folds']) == 2 for r in result.values()))

if __name__ == '__main__': unittest.main()

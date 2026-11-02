import unittest
import numpy as np
from ml.models.classical.folds import fold_features

class FoldTests(unittest.TestCase):
    def test_fold_representations_exclude_held_out_text(self):
        train = [{'title':'alpha common', 'abstract':'shared word'}, {'title':'beta common', 'abstract':'shared word'}]
        held = [{'title':'unseenmarker', 'abstract':'common'}]
        cfg = {'vector_size':8, 'window':2, 'min_count':1, 'epochs':2, 'seed':42}
        f = fold_features(train, held, {}, (1, 2), cfg)
        self.assertNotIn('unseenmarker', f['bow'][2].vocabulary_)
        self.assertNotIn('unseenmarker', f['tfidf'][2].vocabulary_)
        self.assertNotIn('unseenmarker', f['word2vec'][2].wv)
        self.assertEqual(f['tfidf'][0].shape[0], 2)
        self.assertEqual(f['word2vec'][1].shape, (1, 8))
        f2 = fold_features(train, [{'title':'differentmarker', 'abstract':'common'}], {}, (1, 2), cfg)
        np.testing.assert_array_equal(f['tfidf'][2].idf_, f2['tfidf'][2].idf_)
        np.testing.assert_array_equal(f['word2vec'][2].wv.vectors, f2['word2vec'][2].wv.vectors)

if __name__ == '__main__': unittest.main()

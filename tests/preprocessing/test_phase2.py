import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from gensim.models import Word2Vec
from ml.preprocessing.text import TextConfig, document, normalize, tokens
from ml.preprocessing.vocabulary import PAD, UNK, build_vocabulary, choose_length, encode
from ml.representations.features import document_vector, train_word2vec
from ml.representations.phase2 import fit_training, run, transform


def row(title, abstract='common common research'):
    return {'title': title, 'abstract': abstract}

class PreprocessingTests(unittest.TestCase):
    def test_normalize_tokenize_and_options(self):
        self.assertEqual(normalize('  Ａ  B\nC  '), 'a b c')
        self.assertEqual(tokens('École, re-uses!'), ['école', 're', 'uses'])
        self.assertEqual(tokens('The models', TextConfig(stopwords=True)), ['models'])
        self.assertEqual(tokens('running studies', TextConfig(stemming=True)), ['run', 'studi'])
        self.assertEqual(tokens('mice studies', TextConfig(lemmatization=True)), ['mouse', 'study'])
        with self.assertRaises(ValueError):
            TextConfig(stemming=True, lemmatization=True)
        with self.assertRaises(TypeError):
            normalize(None)
        with self.assertRaises(ValueError):
            document({'title': None, 'abstract': 'a'})

    def test_neural_vocabulary_and_sequences(self):
        docs = [['z', 'a', 'a'], ['z', 'b'], []]
        vocab = build_vocabulary(docs, 1, 5)
        self.assertEqual(list(vocab), [PAD, UNK, 'a', 'z', 'b'])
        self.assertEqual(encode(['missing', 'a'], vocab, 4), [1, 2, 0, 0])
        self.assertEqual(encode([], vocab, 2), [0, 0])
        self.assertEqual(encode(['z']*10, vocab, 3), [3, 3, 3])
        self.assertEqual(choose_length(docs, .95), 3)
        self.assertEqual(build_vocabulary(docs, 2), {PAD:0, UNK:1, 'a':2, 'z':3})
        with self.assertRaises(ValueError):
            choose_length([])

    def test_ngram_and_training_only_fitting(self):
        train = [row('apple banana'), row('banana apple')]
        bow, tfidf, vocab, length, model, _ = fit_training(train, TextConfig(), (1,2), 1, 100, .95, {'vector_size':8,'window':2,'min_count':1,'epochs':5,'seed':42})
        self.assertIn('apple banana', bow.vocabulary_)
        before = (dict(bow.vocabulary_), dict(tfidf.vocabulary_), dict(vocab), set(model.wv.key_to_index))
        b, t, seq, vectors = transform([row('validationonly testonly')], bow, tfidf, vocab, length, model)
        self.assertEqual((b.shape[0], t.shape[0], seq.shape, vectors.shape), (1, 1, (1,length), (1,8)))
        self.assertEqual(before, (bow.vocabulary_, tfidf.vocabulary_, vocab, set(model.wv.key_to_index)))
        for word in ('validationonly', 'testonly'):
            self.assertNotIn(word, vocab); self.assertNotIn(word, bow.vocabulary_); self.assertNotIn(word, model.wv)
        self.assertEqual(seq[0, :2].tolist(), [1, 1])
        self.assertTrue(np.array_equal(document_vector(model, []), np.zeros(8)))
        self.assertTrue(np.array_equal(document_vector(model, ['unknownword']), np.zeros(8)))
        np.testing.assert_allclose(document_vector(model, ['apple', 'banana']), (model.wv['apple']+model.wv['banana'])/2)
        model2 = train_word2vec([tokens(document(x)) for x in train], vector_size=8, window=2, min_count=1, epochs=5, seed=42)
        np.testing.assert_array_equal(model.wv.vectors, model2.wv.vectors)

    def test_integration_and_reload(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'phase1'; source.mkdir()
            rows = {'train':[row('apple banana'), row('banana apple')], 'validation':[row('validationonly')], 'test':[row('testonly')]}
            for name, records in rows.items():
                (source/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records), encoding='utf-8')
            (source/'metadata.json').write_text(json.dumps({'source':'fixture','revision':'fixture','split_sizes':{k:len(v) for k,v in rows.items()}}), encoding='utf-8')
            out = root/'phase2'
            w2v = {'vector_size':8,'window':2,'min_count':1,'epochs':3,'seed':42}
            meta = run(source, out, min_frequency=1, w2v=w2v)
            bow = joblib.load(out/'bow.joblib'); tfidf = joblib.load(out/'tfidf.joblib')
            vocab = json.loads((out/'vocabulary.json').read_text(encoding='utf-8'))
            model = Word2Vec.load(str(out/'word2vec.model'))
            result = transform(rows['test'], bow, tfidf, vocab, meta['sequence_length'], model)
            np.testing.assert_array_equal(result[2], np.load(out/'test_sequences.npy'))
            np.testing.assert_allclose(result[3], np.load(out/'test_word2vec.npy'))
            self.assertEqual(result[0].shape, tuple(meta['split_shapes']['test']['bow']))
            self.assertNotIn('testonly', model.wv)
            with self.assertRaises(FileExistsError): run(source, out, w2v=w2v)

    def test_existing_phase1_manifest(self):
        path = Path('artifacts/data/phase1/validation.jsonl')
        self.assertTrue(path.exists())
        with path.open(encoding='utf-8') as f:
            first = json.loads(next(f))
        self.assertTrue(document(first))

if __name__ == '__main__':
    unittest.main()

"""Fit Phase 2 representations exclusively on saved Phase 1 training records."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import gensim
import joblib
import nltk
import numpy as np
import scipy
import sklearn
from scipy.sparse import save_npz
from ml.preprocessing.text import TextConfig, document, tokens
from ml.preprocessing.vocabulary import build_vocabulary, choose_length, encode
from ml.representations.features import document_vector, train_word2vec, vectorizers


def read_manifest(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def fit_training(rows, config, ngrams, min_frequency, max_vocab, quantile, w2v, sequence_length=None):
    if not rows:
        raise ValueError('Training manifest is empty')
    texts = [document(row) for row in rows]
    neural_tokens = [tokens(text) for text in texts]
    vocabulary = build_vocabulary(neural_tokens, min_frequency, max_vocab)
    length = choose_length(neural_tokens, quantile) if sequence_length is None else sequence_length
    if not isinstance(length, int) or length < 1:
        raise ValueError('Sequence length must be a positive integer')
    bow, tfidf = vectorizers(config, ngrams)
    bow.fit(texts)
    tfidf.fit(texts)
    model = train_word2vec(neural_tokens, **w2v)
    return bow, tfidf, vocabulary, length, model, neural_tokens


def transform(rows, bow, tfidf, vocabulary, length, model):
    texts = [document(row) for row in rows]
    tokenized = [tokens(text) for text in texts]
    return (bow.transform(texts), tfidf.transform(texts),
            np.asarray([encode(doc, vocabulary, length) for doc in tokenized], dtype=np.int32).reshape(len(rows), length),
            np.asarray([document_vector(model, doc) for doc in tokenized], dtype=np.float32).reshape(len(rows), model.vector_size))


def run(source, output, config=TextConfig(), ngrams=(1, 2), min_frequency=2, max_vocab=50000, quantile=.95, w2v=None, sequence_length=None):
    source, output = Path(source), Path(output)
    if output.resolve() == source.resolve() or source.resolve() in output.resolve().parents:
        raise ValueError('Output must be outside Phase 1 artifacts')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {output}')
    w2v = w2v or dict(vector_size=100, window=5, min_count=2, epochs=5, seed=42)
    metadata = json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    paths = {name: source/f'{name}.jsonl' for name in ('train', 'validation', 'test')}
    hashes = {name: digest(path) for name, path in paths.items()}
    hashes['metadata'] = digest(source/'metadata.json')
    train = read_manifest(paths['train'])
    if len(train) != metadata['split_sizes']['train']:
        raise ValueError('Training manifest size disagrees with metadata')
    bow, tfidf, vocabulary, length, model, training_tokens = fit_training(train, config, ngrams, min_frequency, max_vocab, quantile, w2v, sequence_length)
    lengths = sorted(map(len, training_tokens))
    summary = {'dataset_source': metadata['source'], 'dataset_revision': metadata['revision'], 'input_sha256': hashes, 'classical_config': asdict(config), 'ngram_range': ngrams, 'neural_config': asdict(TextConfig()), 'vocabulary_min_frequency': min_frequency, 'vocabulary_max_size': max_vocab, 'vocabulary_size': len(vocabulary), 'sequence_quantile': quantile if sequence_length is None else None, 'sequence_length': length, 'training_length_min': lengths[0], 'training_length_median': float(np.median(lengths)), 'training_length_max': lengths[-1], 'training_documents_truncated': sum(n > length for n in lengths), 'word2vec_config': {**w2v, 'workers': 1, 'architecture': 'CBOW', 'aggregation': 'mean of in-vocabulary token vectors; zero if none'}, 'word2vec_vocabulary_size': len(model.wv), 'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'scikit_learn': sklearn.__version__, 'gensim': gensim.__version__, 'nltk': nltk.__version__}, 'split_shapes': {}}
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(bow, output/'bow.joblib')
    joblib.dump(tfidf, output/'tfidf.joblib')
    (output/'vocabulary.json').write_text(json.dumps(vocabulary, ensure_ascii=False, indent=2), encoding='utf-8')
    model.save(str(output/'word2vec.model'))
    for name in paths:
        rows = train if name == 'train' else read_manifest(paths[name])
        if len(rows) != metadata['split_sizes'][name]:
            raise ValueError(f'{name} manifest size disagrees with metadata')
        b, t, seq, vectors = transform(rows, bow, tfidf, vocabulary, length, model)
        save_npz(output/f'{name}_bow.npz', b)
        save_npz(output/f'{name}_tfidf.npz', t)
        np.save(output/f'{name}_sequences.npy', seq)
        np.save(output/f'{name}_word2vec.npy', vectors)
        summary['split_shapes'][name] = {'bow': list(b.shape), 'tfidf': list(t.shape), 'sequences': list(seq.shape), 'word2vec': list(vectors.shape)}
    (output/'metadata.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='artifacts/data/phase1')
    parser.add_argument('--output', default='artifacts/data/phase2')
    parser.add_argument('--stopwords', action='store_true')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--stemming', action='store_true')
    group.add_argument('--lemmatization', action='store_true')
    parser.add_argument('--ngram-max', type=int, default=2)
    parser.add_argument('--min-frequency', type=int, default=2)
    parser.add_argument('--max-vocab', type=int, default=50000)
    parser.add_argument('--quantile', type=float, default=.95)
    parser.add_argument('--sequence-length', type=int, default=None)
    parser.add_argument('--vector-size', type=int, default=100)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--min-count', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    config = TextConfig(stopwords=args.stopwords, stemming=args.stemming, lemmatization=args.lemmatization)
    w2v = dict(vector_size=args.vector_size, window=args.window, min_count=args.min_count, epochs=args.epochs, seed=args.seed)
    print(json.dumps(run(args.source, args.output, config, (1, args.ngram_max), args.min_frequency, args.max_vocab, args.quantile, w2v, args.sequence_length), indent=2))

if __name__ == '__main__':
    main()

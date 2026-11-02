"""Training-only classical and Word2Vec representations."""
from dataclasses import dataclass
import hashlib
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from ml.preprocessing.text import TextConfig, tokens

@dataclass(frozen=True)
class Tokenizer:
    config: TextConfig
    def __call__(self, text):
        return tokens(text, self.config)


def vectorizers(config=TextConfig(), ngram_range=(1, 2)):
    if not 1 <= ngram_range[0] <= ngram_range[1]:
        raise ValueError('Invalid n-gram range')
    kwargs = dict(tokenizer=Tokenizer(config), token_pattern=None, lowercase=False, ngram_range=ngram_range, dtype=np.float32)
    return CountVectorizer(**kwargs), TfidfVectorizer(**kwargs)


def stable_hash(word):
    return int.from_bytes(hashlib.sha256(word.encode('utf-8')).digest()[:8], 'little')


def train_word2vec(train_tokens, vector_size=100, window=5, min_count=2, epochs=5, seed=42):
    if not train_tokens or not any(train_tokens):
        raise ValueError('Word2Vec requires nonempty training tokens')
    return Word2Vec(sentences=train_tokens, vector_size=vector_size, window=window, min_count=min_count, epochs=epochs, seed=seed, workers=1, hashfxn=stable_hash)


def document_vector(model, document_tokens):
    vectors = [model.wv[word] for word in document_tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size, dtype=np.float32)

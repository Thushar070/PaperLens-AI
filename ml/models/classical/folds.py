"""Fold-specific feature construction for classical experiments."""
import numpy as np
from ml.preprocessing.text import TextConfig, document, tokens
from ml.representations.features import vectorizers, train_word2vec, document_vector


def fold_features(train_rows, validation_rows, config, ngrams, word2vec_config):
    train_text = [document(row) for row in train_rows]
    validation_text = [document(row) for row in validation_rows]
    if not train_text:
        raise ValueError('Fold training data is empty')
    bow, tfidf = vectorizers(TextConfig(**config), tuple(ngrams))
    result = {}
    for name, vectorizer in (('bow', bow), ('tfidf', tfidf)):
        result[name] = (vectorizer.fit_transform(train_text), vectorizer.transform(validation_text), vectorizer)
    train_tokens = [tokens(text) for text in train_text]
    validation_tokens = [tokens(text) for text in validation_text]
    model = train_word2vec(train_tokens, **word2vec_config)
    result['word2vec'] = (np.asarray([document_vector(model, doc) for doc in train_tokens]), np.asarray([document_vector(model, doc) for doc in validation_tokens]), model)
    return result

"""Deterministic neural vocabulary and fixed-length encoding."""
from collections import Counter
import math

PAD, UNK = '<PAD>', '<UNK>'


def build_vocabulary(documents, min_frequency=2, max_size=50000):
    if min_frequency < 1 or max_size < 2:
        raise ValueError('min_frequency must be positive and max_size at least 2')
    counts = Counter(token for doc in documents for token in doc if token not in (PAD, UNK))
    ordered = sorted((token for token, count in counts.items() if count >= min_frequency), key=lambda token: (-counts[token], token))
    return {token: i for i, token in enumerate([PAD, UNK] + ordered[:max_size-2])}


def choose_length(documents, quantile=.95):
    lengths = sorted(map(len, documents))
    if not lengths or not 0 < quantile <= 1:
        raise ValueError('A nonempty training set and a quantile in (0, 1] are required')
    return max(1, lengths[math.ceil(len(lengths)*quantile)-1])


def encode(doc, vocabulary, length):
    if length < 1 or vocabulary.get(PAD) != 0 or vocabulary.get(UNK) != 1:
        raise ValueError('Invalid sequence length or special token indices')
    result = [vocabulary.get(token, 1) for token in doc[:length]]
    return result + [0] * (length - len(result))

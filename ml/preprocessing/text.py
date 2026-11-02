"""Reusable text normalization and configurable classical tokenization."""
from dataclasses import dataclass
import re
import unicodedata

from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)

@dataclass(frozen=True)
class TextConfig:
    lowercase: bool = True
    stopwords: bool = False
    stemming: bool = False
    lemmatization: bool = False

    def __post_init__(self):
        if self.stemming and self.lemmatization:
            raise ValueError('Choose stemming or lemmatization, not both')


def normalize(text: str, lowercase: bool = True) -> str:
    if not isinstance(text, str):
        raise TypeError('Text must be a string')
    text = ' '.join(unicodedata.normalize('NFKC', text).split())
    return text.casefold() if lowercase else text


def tokens(text: str, config: TextConfig = TextConfig()) -> list[str]:
    words = TOKEN_PATTERN.findall(normalize(text, config.lowercase))
    if config.stopwords:
        words = [word for word in words if word.casefold() not in ENGLISH_STOP_WORDS]
    if config.stemming:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    if config.lemmatization:
        lemmatizer = WordNetLemmatizer()
        try:
            words = [lemmatizer.lemmatize(word) for word in words]
        except LookupError as exc:
            raise RuntimeError('NLTK WordNet corpus is required for lemmatization') from exc
    return words


def document(row: dict) -> str:
    title, abstract = row.get('title'), row.get('abstract')
    if not isinstance(title, str) or not isinstance(abstract, str):
        raise ValueError('Both title and abstract must be strings')
    return title + ' ' + abstract

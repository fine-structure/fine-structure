from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from .tokenizers import CharTokenizer


def preprocess(text, maxlen, onehot=False):

    tokenizer = CharTokenizer()
    tokenizer.build_vocab(text)
    X = tokenizer.encode(text)

    X = pad_sequences(X, maxlen=maxlen + 1, padding='post', truncating='post')

    y = X[:, 1:]
    X = X[:, :-1]

    y = to_categorical(y)
    if onehot:
        X = to_categorical(X)

    return X, y, tokenizer

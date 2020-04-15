import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


def postprocess(pred, y):

    pred_idx = pred.argmax(axis=-1)
    y_idx = y.argmax(axis=-1)

    errors = (pred_idx != y_idx).astype(float)
    errors[y_idx == 0] = np.nan

    return pred_idx, y_idx, errors


def perplexity(y, pred):

    cross_entropy = []
    for i in range(len(y)):

        pred_i = pred[i]
        y_i = y[i]
        pred_i = pred_i[y_i.argmax(axis=1) != 0]
        y_i = y_i[y_i.argmax(axis=1) != 0]

        cross_entropy.append(log_loss(y_i, pred_i))

    cross_entropy = np.mean(cross_entropy)

    return np.exp(cross_entropy)


def get_char_counts(idx_matrix, idx2char):

    unique, counts = np.unique(idx_matrix, return_counts=True)
    counts = pd.Series(counts, index=unique)
    counts.drop(0, inplace=True)
    counts.index = counts.index.map(idx2char)

    return counts


def get_char_counts_by_position(idx_matrix, idx2char):

    counts_by_position = pd.DataFrame(idx_matrix).apply(
        lambda x: x.value_counts(), axis=0).fillna(0)
    counts_by_position.drop(0, inplace=True)
    counts_by_position.index = counts_by_position.index.map(idx2char)

    return counts_by_position


def generate_sequence(model, seed_sequence, tokenizer,
                      length=50, maxlen=100, onehot=False,
                      temperature=1):

    sequence = list(seed_sequence)
    for i in range(length):

        x = [tokenizer.token2index[tokenizer.bos_token]]
        for token in sequence:
            x.append(tokenizer.encode_token(token))
        x = pad_sequences([x], maxlen=maxlen, truncating='pre')

        if onehot:
            x = to_categorical(x, num_classes=tokenizer.vocab_size)

        preds = model.predict(x, verbose=0)[0][-1]
        next_index = sample_predictions(preds, temperature=temperature)
        next_char = tokenizer.index2token[next_index]
        sequence.append(next_char)

        if next_char == '<eos>':
            break

    return ''.join(sequence)


def sample_predictions(preds, temperature=1.0):
    """Function from https://keras.io/examples/lstm_text_generation."""

    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)

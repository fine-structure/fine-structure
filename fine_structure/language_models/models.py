from tensorflow.keras import optimizers
from tensorflow.keras.layers import (GRU, LSTM, Dense, Embedding, Input,
                                     SpatialDropout1D, TimeDistributed)
from tensorflow.keras.models import Model


def rnn_lm(input_len, vocab_size, rnn_type='GRU',
           embedding_size=None, num_rnn_layers=1,
           num_cells=16, spatial_dropout=0, 
           optimizer='Adam', optimizer_params={'lr': 1e-3},
           loss='categorical_crossentropy', metrics=['accuracy']):

    rnn_layer = get_rnn_layer(rnn_type)

    if embedding_size is None:
        inputs = Input(shape=(input_len, vocab_size), dtype='float32')
        x = inputs
    else:
        inputs = Input(shape=(input_len,), dtype='int32')
        x = Embedding(input_dim=vocab_size, output_dim=embedding_size, 
                      input_length=input_len, trainable=True)(inputs)

    if spatial_dropout > 0:
        x = SpatialDropout1D(spatial_dropout)(x)

    for i in range(num_rnn_layers):
        x = rnn_layer(num_cells, return_sequences=True)(x)

    output = TimeDistributed(Dense(vocab_size, activation='softmax'))(x)

    model = Model(inputs=inputs, outputs=output)
    optimizer = getattr(optimizers, optimizer)(**optimizer_params)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model


def get_rnn_layer(rnn_type):

    if rnn_type == 'GRU':
        return GRU
    elif rnn_type == 'LSTM':
        return LSTM

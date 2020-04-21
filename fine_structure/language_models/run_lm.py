import os

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from fine_structure.language_models.eval import perplexity, postprocess
from fine_structure.language_models.models import rnn_lm
from fine_structure.language_models.preprocess import preprocess
from fine_structure.loggers.mlflow_utils import MlflowLogger
from fine_structure.loggers.set_logger import set_logger
from fine_structure.utils import LoggingCallback


MLFLOW_TRACKING_URI = os.path.join(os.environ['FINE_STRUCTURE_DATA'], 'mlruns')
MLFLOW_EXPERIMENT_NAME = 'first'
LOG_FILE = os.path.join(os.environ['FINE_STRUCTURE_DATA'], 'logs', 
                        os.path.basename(__file__).split('.')[0] + '.log')


config = {}
config['data_path'] = 'tox21/tox21.csv'
config['max_len'] = 100
config['model_params'] = {'rnn_type': 'GRU',
                          'embedding_size': None,
                          'num_rnn_layers': 1,
                          'num_cells': 16,
                          'spatial_dropout': 0,
                          'optimizer': 'Adam',
                          'optimizer_params': {'lr': 1e-2}
                          }
config['train_params'] = {'batch_size': 32,
                          'epochs': 10,
                          'validation_split': 0.1}
config['early_stopping_patience'] = 5


df = pd.read_csv(os.path.join(os.environ['FINE_STRUCTURE_DATA'], config['data_path']))
texts = df.smiles.tolist()


def run_experiment(config):
    """Run experiment."""

    logger = set_logger(logger_name=__name__, log_file=LOG_FILE)
    mlflow_logger = MlflowLogger(experiment_name=MLFLOW_EXPERIMENT_NAME,
                                 tracking_uri=MLFLOW_TRACKING_URI,
                                 log_file=LOG_FILE,
                                 source_file=os.path.abspath(__file__))

    with mlflow.start_run():

        mlflow_logger.log_config(config)
        mlflow_logger.set_tags(tags={'type': 'single_run'})

        if config['model_params']['embedding_size'] is None:
            X, y, tokenizer = preprocess(texts, maxlen=config['max_len'], onehot=True)
        else:
            X, y, tokenizer = preprocess(texts, maxlen=config['max_len'])
        logger.info('vocab size: %s', tokenizer.vocab_size)
        logger.info('X shape: %s, y shape: %s', X.shape, y.shape)

        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)
        logger.info('Xtrain shape: %s, xtest shape: %s', Xtrain.shape, Xtest.shape)

        model = rnn_lm(input_len=ytrain.shape[1], vocab_size=ytrain.shape[2],
                       **config['model_params'])
        logger.info(model.summary(print_fn=logger.info))

        early_stopping = EarlyStopping(monitor='val_loss', mode='auto',
                                       verbose=1, patience=config['early_stopping_patience'],
                                       restore_best_weights=True)
        if config['model_params']['embedding_size'] is None:
            model.fit(Xtrain, ytrain, **config['train_params'],
                      callbacks=[LoggingCallback(logger.info), early_stopping])
        else:
            model.fit(Xtrain, ytrain, **config['train_params'],
                      callbacks=[LoggingCallback(logger.info), early_stopping])
        pred = model.predict(Xtest)

        pred_idx, ytest_idx, errors = postprocess(pred, ytest)
        metrics = {'accuracy': 1 - np.nanmean(errors),
                   'perplexity': perplexity(ytest, pred)}
        logger.info('Mean accuracy: %.3f', metrics["accuracy"])
        logger.info('Perplexity: %.3f', metrics["perplexity"])

        mlflow_logger.log_metrics(metrics)
        mlflow_logger.log_artifacts()
        mlflow_logger.log_keras_models(model)


run_experiment(config)

# config['max_len'] = 50
# run_experiment(config)

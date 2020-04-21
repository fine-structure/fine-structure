"""Mlflow logger."""

import os
import shutil
import tempfile

import mlflow
import pandas as pd


TEMP_DIR = os.path.join(os.path.expanduser("~"), 'tmp')


class MlflowLogger():

    def __init__(self, experiment_name, tracking_uri,
                 log_file=None, source_file=None,
                 source_dir=None, flatten_params=True,
                 short_id_len=4):

        self._experiment_name = experiment_name
        self._tracking_uri = tracking_uri
        self.log_file = log_file
        self.source_file = source_file
        self.source_dir = source_dir
        self.flatten_params = flatten_params
        self.short_id_len = short_id_len

        self.set()

    def set(self):

        mlflow.set_tracking_uri(self._tracking_uri)
        mlflow.set_experiment(self._experiment_name)

    def log_config(self, config):

        self.log_params(config)

        with tempfile.TemporaryDirectory(dir=TEMP_DIR) as tmpdir:
            path = os.path.join(tmpdir, 'config.yml')
            to_yaml(config, path)
            mlflow.log_artifact(path)

    def log_params(self, parameters):

        if self.flatten_params:
            parameters = flatten(parameters, sep='.')
        for param_name, param_value in parameters.items():
            mlflow.log_param(param_name, param_value)

    def log_metrics(self, metrics):

        if isinstance(metrics, pd.Series):
            metrics = metrics.to_dict()

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

    def log_artifacts(self, additional_files=None):

        if self.source_file is not None:
            mlflow.log_artifact(self.source_file)

        if self.source_dir is not None:
            with tempfile.TemporaryDirectory(dir=TEMP_DIR) as tmpdir:
                path = os.path.join(tmpdir, 'source')
                shutil.make_archive(path, 'zip', self.source_dir)
                mlflow.log_artifact(path + '.zip')

        if self.log_file is not None:
            mlflow.log_artifact(self.log_file)

        if additional_files is not None:
            for f in additional_files:
                mlflow.log_artifact(f)

    def log_dataframes(self, dataframes):

        short_run_id = mlflow.active_run().info.run_id[-self.short_id_len:]

        with tempfile.TemporaryDirectory(dir=TEMP_DIR) as tmpdir:
            for name, df in dataframes.items():
                path = os.path.join(tmpdir, f'{name}_{short_run_id}.csv')
                df.to_csv(path)
                mlflow.log_artifact(path)

    def log_keras_models(self, models, save_only_weights=True):

        short_run_id = mlflow.active_run().info.run_id[-self.short_id_len:]
        if not isinstance(models, list):
            models = [models]

        with tempfile.TemporaryDirectory(dir=TEMP_DIR) as tmpdir:
            for i, model in enumerate(models):
                path = os.path.join(tmpdir, f'model_{short_run_id}_{i}.h5')
                if save_only_weights:
                    model.save_weights(path)
                else:
                    model.save(path)
                mlflow.log_artifact(path)

    def set_tags(self, tags=None, comment=None):

        mlflow.set_tag('short_id', mlflow.active_run().info.run_id[-self.short_id_len:])
        if comment:
            mlflow.set_tag('mlflow.note.content', comment)
        if tags:
            mlflow.set_tags(tags)


def to_yaml(obj, path):

    with open(path, 'w') as file:
        yaml.dump(obj, file)


def flatten(d, parent_key='', sep='.'):
    """
    Flatten nested dictionary
    source: 
    https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys

    alternative: pd.io.json.json_normalize(d, sep='.')
    """

    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def inflate(d, sep="."):
    """
    Inflate flattened nested dictionary back
    source: https://gist.github.com/fmder/494aaa2dd6f8c428cede
    """

    items = dict()
    for k, v in d.items():
        keys = k.split(sep)
        sub_items = items
        for ki in keys[:-1]:
            try:
                sub_items = sub_items[ki]
            except KeyError:
                sub_items[ki] = dict()
                sub_items = sub_items[ki]

        sub_items[keys[-1]] = v

    return items
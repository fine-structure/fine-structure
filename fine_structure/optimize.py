"""
Parameters optimization with hyperopt.
"""

import logging
import timeit
from functools import partial

import pandas as pd

from hyperopt import STATUS_OK, Trials, fmin, tpe


def run_hyperopt(loss_function, data, fspace, num_iterations=20,
                 algo=tpe.suggest, n_startup_jobs=None,
                 log_file='hyperopt_log.csv'):
    """
    Run parameters optimization using hyperopt.

    Args:
        loss_function: function to optimize
            inputs - data, params
            outputs - loss, metrics
        data: some object containing data
        fspace: hyperopt search space
        num_iterations (int): number of iterations
        algo: hyperopt optimization algorithm
            options - hyperopt.tpe.suggest, hyperopt.rand.suggest
        n_startup_jobs (int): define number of first random iterations for tpe algorithm
        log_file (str): filename for saving dataframe with results

    Returns:
        trials: hyperopt Trials object
        results: pd.DataFrame with optimization results
    """

    results = []

    # objective function to pass to hyperopt
    def objective(params):

        iteration_start = timeit.default_timer()
        logging.info(params)

        res = loss_function(data, params)
        loss, metrics = res[0], res[1]

        iteration_time = timeit.default_timer() - iteration_start
        logging.info('iteration time %.1f, loss %.5f' % (iteration_time, loss))

        metrics['n_iter'] = len(results)
        metrics['time'] = iteration_time

        results.append(metrics)
        intermediate_results = pd.concat(results)[results[0].columns]
        intermediate_results.to_csv(log_file, index=False)

        return {'loss': loss, 'status': STATUS_OK,
                'runtime': iteration_time,
                'params': params, 'metrics': metrics}

    # object with history of iterations to pass to hyperopt
    trials = Trials()

    # run hyperopt
    if n_startup_jobs is not None:
        best = fmin(fn=objective, space=fspace,
                    algo=partial(algo, n_startup_jobs=n_startup_jobs),
                    max_evals=num_iterations, trials=trials)
    else:
        best = fmin(fn=objective, space=fspace,
                    algo=algo, max_evals=num_iterations, trials=trials)

    results = pd.concat(results)[results[0].columns]
    logging.info('finished.')
    logging.info('best parameters', trials.best_trial['result']['params'])

    return trials, results

import logging

logging.basicConfig(level=logging.WARNING)

import hpbandster.core.nameserver as hpns

from hpbandster.optimizers import RandomSearch

import ConfigSpace as CS
from hpbandster.core.worker import Worker
import argparse

from cnn_mnist import mnist, train_and_validate


class MyWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test = mnist("./")

    def compute(self, config, budget, **kwargs):
        """
        Evaluates the configuration on the defined budget and returns the validation performance.

        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train
        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """
        lr = config["learning_rate"]
        num_filters = config["num_filters"]
        batch_size = config["batch_size"]
        filter_size = config["filter_size"]

        epochs = budget

        # TODO: train and validate your convolutional neural networks here
        model, learning_curve= train_and_validate(self.x_train, self.y_train, self.x_valid, self.y_valid, int(budget),
                                                  lr, num_filters, batch_size, filter_size)

        validation_error = 1 - learning_curve["val_acc"][-1]

        # TODO: We minimize so make sure you return the validation error here
        return ({
            'loss': validation_error,  # this is the a mandatory field to run hyperband
            'info': {}  # can be used for any user-defined information - also mandatory
        })

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()

        # TODO: Implement configuration space here. See https://github.com/automl/HpBandSter/blob/master/hpbandster/examples/example_5_keras_worker.py  for an example
        lr = CSH.UniformFloatHyperparameter('learning_rate', lower=1e-4, upper=1e-1, default_value='1e-3', log=True)
        batch_size = CSH.UniformIntegerHyperparameter('batch_size', lower=16, upper=128, default_value='128', log=True)
        num_filters = CSH.UniformIntegerHyperparameter('num_filters', lower=2e3, upper=2e6, default_value='2e5', log=True)
        filter_size = CSH.CategoricalHyperparameter('filter_size', [3, 5])

        config_space.add_hyperparameters([lr, batch_size, num_filters, filter_size])

        return config_space


parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--budget', type=float,
                    help='Maximum budget used during the optimization, i.e the number of epochs.', default=12)
parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=20)
args = parser.parse_args()

# Step 1: Start a nameserver
# Every run needs a nameserver. It could be a 'static' server with a
# permanent address, but here it will be started for the local machine with the default port.
# The nameserver manages the concurrent running workers across all possible threads or clusternodes.
# Note the run_id argument. This uniquely identifies a run of any HpBandSter optimizer.
NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
NS.start()

# Step 2: Start a worker
# Now we can instantiate a worker, providing the mandatory information
# Besides the sleep_interval, we need to define the nameserver information and
# the same run_id as above. After that, we can start the worker in the background,
# where it will wait for incoming configurations to evaluate.
w = MyWorker(nameserver='127.0.0.1', run_id='example1')
w.run(background=True)

# Step 3: Run an optimizer
# Now we can create an optimizer object and start the run.
# Here, we run RandomSearch, but that is not essential.
# The run method will return the `Result` that contains all runs performed.

rs = RandomSearch(configspace=w.get_configspace(),
                  run_id='example1', nameserver='127.0.0.1',
                  min_budget=int(args.budget), max_budget=int(args.budget))
res = rs.run(n_iterations=args.n_iterations)

# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
rs.shutdown(shutdown_workers=True)
NS.shutdown()

# Step 5: Analysis
# Each optimizer returns a hpbandster.core.result.Result object.
# It holds information about the optimization run like the incumbent (=best) configuration.
# For further details about the Result object, see its documentation.
# Here we simply print out the best config and some statistics about the performed runs.
id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()

print('Best found configuration:', id2config[incumbent]['config'])


# Plots the performance of the best found validation error over time
all_runs = res.get_all_runs()
# Let's plot the observed losses grouped by budget,
import hpbandster.visualization as hpvis

hpvis.losses_over_time(all_runs)

import matplotlib.pyplot as plt
plt.savefig("random_search.png")

# TODO: retrain the best configuration (called incumbent) and compute the test error
best_config = id2config[incumbent]['config']
model, _ = train_and_validate(w.x_train, w.y_train, w.x_valid, w.y_valid, int(args.budget),
                              best_config["learning_rate"], best_config["num_filters"],
                              best_config["batch_size"], beat_config["filter_size"])

_, test_acc = model.evaluate(w.x_test, w.y_test)

print('Test error:', 1 - test_acc)
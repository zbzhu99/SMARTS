import os

# Set pythonhashseed
os.environ["PYTHONHASHSEED"] = "0"
# Silence the logs of TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
import numpy as np

np.random.seed(123)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
import random as python_random

python_random.seed(123)

# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
import tensorflow as tf

tf.random.set_seed(123)

# --------------------------------------------------------------------------

import dreamerv2.api as dv2
import warnings
import yaml

from examples.auto_drive.env import single_agent
from datetime import datetime
from pathlib import Path


def main(config):

    # Create env
    print("[INFO] Creating environments")
    env = single_agent.SingleAgent(config, config["seed"])

    time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    logdir = (Path(__file__).absolute().parent).joinpath("logs").joinpath(f"{time}")

    config = dv2.defaults.update({
        'logdir': logdir,
        'log_every': 1e3,
        'train_every': 10,
        'prefill': 1e5,
        'actor_ent': 3e-3,
        'loss_scales.kl': 1.0,
        'discount': 0.99,
    }).parse_flags()

    # Train dreamerv2 with env
    dv2.train(env, config)


if __name__ == "__main__":
    config_yaml = (Path(__file__).absolute().parent).joinpath("config.yaml")
    with open(config_yaml, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Setup GPU
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        warnings.warn(
            f"Not configured to use GPU or GPU not available.",
            ResourceWarning,
        )
        # raise SystemError("GPU device not found")

    main(config=config["dreamerv2"])

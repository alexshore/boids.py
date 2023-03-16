import numpy as np
import tensorflow as tf

import time


def convert_weights_to_network(weights) -> tf.keras.models.Sequential:
    """

    input should be the following:
    - current coordinates (2)
    - current direction (1)
    - distance to closest boids (3)
    - angle to closest boids (3)
    - difference in direction to closest boids (3)
    - distance to closest obstacle (1)
    - angle to closest obstacle (1)

    input should be of shape (2 + 1 + 3 + 3 + 3 + 1 + 1,) = (14,)

    weights will be one flat list so need to convert to shape
    (14, 28) + (28, 14) + (14, 7) + (7, 2)

    sections of the list will be:
     0 - 391,
     392 - 783,
     784 - 881,
     882 - 896,


    """
    network = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(14,)),
            tf.keras.layers.Dense(14, activation="relu", use_bias=False),
            tf.keras.layers.Dense(7, activation="relu", use_bias=False),
            tf.keras.layers.Dense(1, activation="tanh", use_bias=False),
        ]
    )
    layered_weights = [
        np.reshape(weights[:196], (14, 14)),
        np.reshape(weights[196:294], (14, 7)),
        np.reshape(weights[294:301], (7, 1)),
    ]
    network.set_weights(layered_weights)
    return network


def main():
    network = convert_weights_to_network(np.random.normal(0, 0.5, 301))
    result = network.predict([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])[0][0]
    print(result)


if __name__ == "__main__":
    main()

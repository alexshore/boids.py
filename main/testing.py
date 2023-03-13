import tensorflow as tf
from numpy import array, float32, random, ndarray
import numpy as np
import time


# def create_model():
#     model = tf.keras.Sequential(
#         [
#             tf.keras.layers.Input(shape=(4,)),
#             # tf.keras.layers.Dense(26),
#             # tf.keras.layers.Dense(10),
#             tf.keras.layers.Dense(3, use_bias=False),
#             tf.keras.layers.Dense(2, use_bias=False, activation="tanh"),
#         ]
#     )
#     return model

start_time = time.time()
print("starting...")
weights = [random.normal(0.0, 0.5, (4, 3)), random.normal(0.0, 0.5, (3, 2))]
print(weights)
flat_weights = np.concatenate((np.ndarray.flatten(weights[0]), np.ndarray.flatten(weights[1])))
print(flat_weights)
weights = [np.reshape(flat_weights[:12], (4, 3)), np.reshape(flat_weights[12:], (3, 2))]
print(weights)

print(f"time taken {time.time() - start_time}")

# np.reshape()

# print(weights)

# model = create_model()

# # print(model.get_weights())


# # weights = [
# #     array([[-0.15190649, 0.57557535, 0.20143974], [-0.9153545, 0.38251483, -0.46655238], [0.423164, 0.5199772, -0.20297432], [-0.18541771, -0.7244631, 0.5278269]]),
# #     array([[-0.16750634, -1.0621506], [0.5835178, -0.6196396], [0.16478908, -0.61518]]),
# # ]

# model.set_weights(weights)

# print(model.get_weights())

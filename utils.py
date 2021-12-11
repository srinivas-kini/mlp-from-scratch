# utils.py: Utility file for implementing helpful utility functions used by the ML algorithms.
#
# Submitted by: Srinivas Kini - skini
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff
import numpy as np

np.seterr(divide='ignore', invalid='ignore')


def euclidean_distance(x1, x2):
    return np.linalg.norm(x1 - x2)


def manhattan_distance(x1, x2):
    return np.sum(np.array([np.abs(i - j) for i, j in zip(x1, x2)]))


def identity(x, derivative=False):
    return np.ones(shape=x.shape, dtype=float) if derivative else x


def sigmoid(x, derivative=False):
    x = np.clip(x, -1e10, 1e10)
    fx = 1.0 / (1.0 + np.exp(-x))
    return fx * (1.0 - fx) if derivative else fx


def tanh(x, derivative=False):
    fx = np.tanh(x)
    return (1.0 - np.square(fx)) if derivative else fx


def relu(x, derivative=False):
    return 1.0 * (x > 0) if derivative else x * (x > 0)


def softmax(x, derivative=False):
    x = np.clip(x, -1e100, 1e100)
    if not derivative:
        c = np.max(x, axis=1, keepdims=True)
        return np.exp(x - c - np.log(np.sum(np.exp(x - c), axis=1, keepdims=True)))
    else:
        return softmax(x) * (1 - softmax(x))


def cross_entropy(y, p):
    return -np.sum(np.sum(y * np.log(p), axis=1)) / y.shape[0]


def one_hot_encoding(y):
    labels = set(y)
    encoding_lookup = {label: num_code for num_code, label in
                       enumerate(sorted(labels))}  # Assign the labels a numeric code
    one_hot_encoded = np.zeros(shape=(len(y), len(labels)), dtype=int)
    yT = np.array([y]).T  # Convert to shape=(n_samples,1)

    for idx, row in enumerate(yT):
        one_hot_encoded[idx, encoding_lookup[row[0]]] = 1

    return one_hot_encoded

# multilayer_perceptron.py: Machine learning implementation of a Multilayer Perceptron classifier from scratch.
#
# Submitted by: Srinivas Kini - skini
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

import numpy as np
from utils import identity, sigmoid, tanh, relu, softmax, cross_entropy, one_hot_encoding


class MultilayerPerceptron:

    def __init__(self, n_hidden=16, hidden_activation='sigmoid', n_iterations=1000, learning_rate=0.01):
        # Create a dictionary linking the hidden_activation strings to the functions defined in utils.py
        activation_functions = {'identity': identity, 'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu}

        # Check if the provided arguments are valid
        if not isinstance(n_hidden, int) \
                or hidden_activation not in activation_functions \
                or not isinstance(n_iterations, int) \
                or not isinstance(learning_rate, float):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the MultilayerPerceptron model object
        self.n_hidden = n_hidden
        self.hidden_activation = activation_functions[hidden_activation]
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self._output_activation = softmax
        self._loss_function = cross_entropy
        self._loss_history = []
        self._X = None
        self._y = None
        self._h_weights = None
        self._h_bias = None
        self._o_weights = None
        self._o_bias = None
        self.hidden_weighted_sum = None
        self.hidden_layer_activation = None
        self.output_weighted_sum = None
        self.mlp_output = None

    def _initialize(self, X, y):
        self._X = X
        self._y = one_hot_encoding(y)
        np.random.seed(42)

        n_samples, n_features = X.shape
        n_hidden, n_outputs = self.n_hidden, len(set(y))

        # Hidden Layer
        self._h_weights = np.random.rand(n_features, n_hidden)
        self._h_bias = np.ones(shape=(1, n_hidden), dtype=float)

        # Output Layer
        self._o_weights = np.random.rand(n_hidden, n_outputs)
        self._o_bias = np.ones(shape=(1, n_outputs), dtype=float)

    def fit(self, X, y):
        self._initialize(X, y)

        # Train the MLP
        for i in range(self.n_iterations):
            self.feed_forward(self._X)
            self.backpropagate()
            if i % 20 == 0:
                loss = self._loss_function(self._y, self.mlp_output)
                self._loss_history.append(loss)

    def feed_forward(self, data):
        # Hidden Layer activation
        self.hidden_weighted_sum = np.dot(data, self._h_weights) + self._h_bias  # u = X.wh + b
        self.hidden_layer_activation = self.hidden_activation(self.hidden_weighted_sum)  # g(u) - hidden

        # Output Layer Activation
        self.output_weighted_sum = np.dot(self.hidden_layer_activation, self._o_weights) + self._o_bias  # u = Z.wo + b
        self.mlp_output = self._output_activation(self.output_weighted_sum)  # output of the MLP

    def backpropagate(self):
        lr = self.learning_rate

        # Calculate errors in the output layer
        output_error = self._y - self.mlp_output
        output_slope = self._output_activation(self.output_weighted_sum, derivative=True)
        output_delta = np.multiply(output_error, output_slope)

        # Calculate errors in the hidden layer
        hidden_error = np.dot(output_delta, self._o_weights.T)
        hidden_slope = self.hidden_activation(self.hidden_weighted_sum, derivative=True)
        hidden_delta = np.multiply(hidden_error, hidden_slope)

        # Update weights
        self._o_weights += np.dot(self.hidden_layer_activation.T, output_delta) * lr
        self._h_weights += np.dot(self._X.T, hidden_delta) * lr

        # Update biases
        self._h_bias += np.sum(hidden_delta, axis=0) * lr
        self._o_bias += np.sum(output_delta, axis=0) * lr

    def predict(self, X):
        self.feed_forward(X)
        return np.array([np.argmax(x) for x in self.mlp_output])

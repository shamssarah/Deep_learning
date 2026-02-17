import logging
import math
import numpy as np
import torch
from copy import deepcopy
from collections import OrderedDict

from .activations import relu, softmax, cross_entropy


class AutogradNeuralNetwork:
    """Implementation that uses torch.autograd

        Neural network classifier with cross-entropy loss
        and ReLU activations
    """
    def __init__(self, shape, gpu_id=-1):
        """Initialize the network

        Args:
            shape: a list of integers that specifieds
                    the number of neurons at each layer.
            gpu_id: -1 means using cpu. 
        """
        self.shape = shape
        # declare weights and biases
        if gpu_id == -1:
            self.weights = [torch.nn.Parameter(torch.FloatTensor(j, i),
                                requires_grad=True)
                            for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.biases = [torch.nn.Parameter(torch.FloatTensor(i, 1),
                                requires_grad=True)
                           for i in self.shape[1:]]
        else:
            self.weights = [torch.nn.Parameter(torch.randn(j, i).cuda(gpu_id),
                                requires_grad=True)
                            for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.biases = [torch.nn.Parameter(torch.randn(i, 1).cuda(gpu_id),
                                requires_grad=True)
                           for i in self.shape[1:]]
        # initialize weights and biases
        self.init_weights()

    def init_weights(self):
        """Initialize weights and biases

            Initialize self.weights and self.biases with
            Gaussian where the std is 1 / sqrt(n_neurons)
        """
        raise NotImplementedError # TODO: Implement this

    def _feed_forward(self, X):
        """Forward pass

        Args:
            X: (n_neurons, n_examples)

        Returns:
            (outputs, act_outputs).

            "outputs" is a list of torch tensors. Each tensor is the Wx+b (weighted sum plus bias)
            of each layer in the shape (n_neurons, n_examples).

            "act_outputs" is also a list of torch tensors. Each tensor is the "activated" outputs
            of each layer in the shape(n_neurons, n_examples). If f(.) is the activation function,
            this should be f(ouptuts).
        """
        raise NotImplementedError # TODO: Implement this

    def train_one_epoch(self, X, y, y_1hot, learning_rate):
        """Train for one epoch

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss for the given input
        """
        X_t = X.t()
        y_1hot_t = y_1hot.t()

        X_t_train = X_t
        y_1hot_t_train = y_1hot_t
        
        # feed forward
        outputs, act_outputs = self._feed_forward(X_t_train)
        loss = cross_entropy(act_outputs[-1], y_1hot_t_train)

        # backward
        loss.backward()

        # update weights and biases
        raise NotImplementedError # TODO: Implement this
    
    def loss(self, X, y, y_1hot):
        """Compute feed forward loss

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss for the given input
        """
        X_t = X.t()
        y_1hot_t = y_1hot.t()
        outputs, act_outputs = self._feed_forward(X_t)
        loss = cross_entropy(act_outputs[-1], y_1hot_t)
        return loss.item()

    def predict(self, X):
        """Predict

            Make predictions for X using the current model

        Args:
            X: (n_examples, n_features)

        Returns:
            (n_examples, )
        """
        outputs, act_outputs = self._feed_forward(X.t())
        # Return class predictions
        raise NotImplementedError # TODO: Implement this


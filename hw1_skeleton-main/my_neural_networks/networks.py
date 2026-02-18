import logging
import math
import numpy as np
import torch
from copy import deepcopy
from collections import OrderedDict

from .activations import relu, softmax, cross_entropy, stable_softmax


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

        device = self.weights[0].device if self.weights else torch.device('cpu')
        # mean is assumed as 0 as it is a gaussian distribution

        self.weights = []
        self.biases = []

        for i in range(len(self.shape) - 1):
            n_neurons_in = self.shape[i]
            n_neurons_out = self.shape[i + 1]

            std = 1 / math.sqrt(n_neurons_in)

            weight = torch.randn(n_neurons_out, n_neurons_in, device=device) * std
            weight.requires_grad = True
            
            bias = torch.zeros(n_neurons_out, 1, device=device)
            bias.requires_grad = True

            self.weights.append(torch.nn.Parameter(weight, requires_grad=True))
            self.biases.append(torch.nn.Parameter(bias, requires_grad=True))




        # self.weights = [torch.nn.Parameter(
        #                     torch.randn(i,j,dtype = torch.float32, requires_grad=True)/math.sqrt(j))
        #                 for i, j in zip(self.shape[1:], self.shape[:-1])]

        
        # self.biases = [torch.nn.Parameter(
        #                     torch.zeros(i,1,dtype = torch.float32, requires_grad=True))
        #         for i in self.shape[1:]]


        # raise NotImplementedError # TODO: Implement this

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
        outputs = []
        act_outputs = []

        for i in range(len(self.weights)):
            output = self.weights[i] @ X + self.biases[i]
            outputs.append(output)

            if i < len(self.weights) - 1:
                X = relu(output)
            else:
                X = stable_softmax(output)
            
            act_outputs.append(X)

        return outputs, act_outputs

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

        with torch.no_grad():
            for i in range(len(self.weights)):
                self.weights[i].data -= learning_rate * self.weights[i].grad.data
                self.biases[i].data -= learning_rate * self.biases[i].grad.data
                self.weights[i].grad.zero_()
                self.biases[i].grad.zero_()
        return loss.item()

        # raise NotImplementedError # TODO: Implement this
    
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

        return torch.argmax(act_outputs[-1], dim=0)
        # raise NotImplementedError # TODO: Implement this


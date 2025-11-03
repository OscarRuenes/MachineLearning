'''
Author: Lalith Vennapusa, Oscar Ruenes
Date: 10/30/2025
Title: CS 4375 Assignment 4 Neural Networks
Desc: Neural network implementation trained on given learning rate, training/testing data, 
number of iterations, and number of hidden layers and hidden nodes.
'''

import argparse
import math
from typing import List

parser = argparse.ArgumentParser()
parser.add_argument("train_path", type=str)
parser.add_argument("test_path", type=str)
parser.add_argument("num_hidden_layers", type=int)
parser.add_argument("num_hidden_nodes", type=int)
parser.add_argument("learning_rate", type=float)
parser.add_argument("num_iterations", type=int)
args = parser.parse_args()

def load_data(path: str):
    try:
        with open(path, 'r') as file:
            lines = [line.strip().split('\t') for line in file if line.strip()]
        headers = lines[0][:-1]
        data = [[float(x) for x in line[:-1]] for line in lines[1:]]
        classes = [float(line[-1]) for line in lines[1:]]
        return headers, data, classes
    except Exception as e:
        print(f"Error loading data from {path}: {e}")
        exit(1)

# get training and testing data, labels in input_labels, train_x has features, train_y has classifications
input_labels, train_X, train_y = load_data(args.train_path)
_, test_X, test_y = load_data(args.test_path)

num_inputs = len(input_labels)

class NeuralNetworkNode:
    def __init__(self, num_inputs: int):
        # +1 for bias
        self.weights = [0.0 for _ in range(num_inputs + 1)]

    def activate(self, inputs: List[float]) -> float:
        # +1 for bias
        inputs = inputs + [1.0]
        z = sum(w * x for w, x in zip(self.weights, inputs))
        return 1.0 / (1.0 + math.exp(-z))

    def update_weights(self, inputs: List[float], delta: float, learning_rate: float):
        inputs = inputs + [1.0]
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * delta * inputs[i]

class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden_layers, num_hidden_nodes, learning_rate):
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_nodes = num_hidden_nodes
        self.learning_rate = learning_rate

        # Create layers
        self.layers: List[List[NeuralNetworkNode]] = []
        prev_size = num_inputs
        for _ in range(num_hidden_layers):
            layer = [NeuralNetworkNode(prev_size) for _ in range(num_hidden_nodes)]
            self.layers.append(layer)
            prev_size = num_hidden_nodes
        # Output layer (1 node)
        self.layers.append([NeuralNetworkNode(prev_size)])

    def forward_pass(self, x: List[float]) -> List[List[float]]:
        activations = [x]
        for layer in self.layers:
            layer_act = [node.activate(activations[-1]) for node in layer]
            activations.append(layer_act)
        return activations

    def back_propagation(self, activations: List[List[float]], y_actual: float):
        deltas = []

        output = activations[-1][0]
        delta_output = (output - y_actual) * output * (1 - output)
        deltas.append([delta_output])

        # Hidden layer deltas (backwards)
        for l in reversed(range(self.num_hidden_layers)):
            layer = self.layers[l]
            next_layer = self.layers[l + 1]
            next_delta = deltas[0]
            delta = []
            for i, node in enumerate(layer):
                currSum = sum(next_delta[k] * next_layer[k].weights[i] for k in range(len(next_layer)))
                delta_i = currSum * activations[l+1][i] * (1 - activations[l+1][i])
                delta.append(delta_i)
            deltas.insert(0, delta)

        # weight updates
        for l, layer in enumerate(self.layers):
            for i, node in enumerate(layer):
                node.update_weights(activations[l], deltas[l][i], self.learning_rate)

def avg_squared_error(nn: NeuralNetwork, X: List[List[float]], y: List[float]) -> float:
    error_sum = 0.0
    for xi, yi in zip(X, y):
        output = nn.forward_pass(xi)[-1][0]
        error_sum += (yi - output) ** 2
    return error_sum / len(y)

def main():
    if len(vars(args)) != 6:
        print("Invalid number of arguments, use train_path, test_path, num_hidden_layers, num_hidden_nodes, learning_rate, num_iterations")
        exit(1)
    nn = NeuralNetwork(num_inputs, args.num_hidden_layers, args.num_hidden_nodes, args.learning_rate)

    num_train = len(train_X)
    num_test = len(test_X)

    for i in range(args.num_iterations):
        idx = i % num_train
        x_instance = train_X[idx]
        y_instance = train_y[idx]

        activations = nn.forward_pass(x_instance)
        nn.back_propagation(activations, y_instance)

        forward_output = activations[-1][0]
        train_error = avg_squared_error(nn, train_X, train_y)
        test_error = avg_squared_error(nn, test_X, test_y)

        print(f"In iteration {i+1}:")
        print(f"Forward pass output: {forward_output:.4f}")
        print(f"Average squared error on training set ({num_train} instances): {train_error:.4f}")
        print(f"Average squared error on test set ({num_test} instances): {test_error:.4f}\n")
main()
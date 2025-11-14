'''
Group Members: Lalith Vennapusa, Oscar Ruenes
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

input_labels, train_X, train_y = load_data(args.train_path)
_, test_X, test_y = load_data(args.test_path)

num_inputs = len(input_labels)

class NeuralNetworkNode:
    def __init__(self, num_inputs: int):
        self.num_inputs = num_inputs
        self.weights = [0.0 for _ in range(num_inputs + 1)]

    def activate(self, inputs: List[float]) -> float:
        inputs = inputs + [1.0]
        assert len(inputs) == len(self.weights)
        z = sum(w * x for w, x in zip(self.weights, inputs))
        return 1.0 / (1.0 + math.exp(-z))

    def update_weights(self, inputs: List[float], delta: float, learning_rate: float):
        inputs = inputs + [1.0]
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * delta * inputs[i]

class NeuralNetworkLayer:
    def __init__(self, num_nodes: int, num_inputs: int):
        self.nodes: List[NeuralNetworkNode] = [NeuralNetworkNode(num_inputs) for _ in range(num_nodes)]
        self.activations: List[float] = [0.0 for _ in range(num_nodes)]
        self.deltas: List[float] = [0.0 for _ in range(num_nodes)]
        self.num_nodes = num_nodes

    def process_input(self, input: List[float]):
        for i in range(0, len(self.activations)):
            self.activations[i] = self.nodes[i].activate(input)

    def update_layer(self, inputs: List[float], learning_rate):
        for i in range(0, len(self.activations)):
            self.nodes[i].update_weights(inputs, self.deltas[i], learning_rate)


    
class NeuralNetwork: 
    def __init__(self, num_inputs: int, num_hidden_layers: int, num_hidden_nodes: int, learning_rate: float):
        self.num_layers: int = num_hidden_layers+2
        self.num_hidden_nodes: int = num_hidden_nodes
        self.learning_rate: float = learning_rate
        self.num_inputs: int = num_inputs

        self.layers: List[NeuralNetworkLayer] = [NeuralNetworkLayer(self.num_inputs, 0)]

        prev_size = num_inputs
        for _ in range(num_hidden_layers):
            layer = NeuralNetworkLayer(num_hidden_nodes, prev_size)
            self.layers.append(layer)
            prev_size = num_hidden_nodes

        self.layers.append(NeuralNetworkLayer(1, prev_size))

    def get_output(self)-> float:
        return self.layers[-1].activations[0]

    def forward_pass(self, input: List[float]) -> float:
        assert len(input) == self.layers[0].num_nodes
        self.layers[0].activations = input
        for i in range(1, len(self.layers)):
            self.layers[i].process_input(self.layers[i-1].activations)
        return self.get_output()
    
    def output_back_prop(self, y_actual: float):
        activation: float = self.get_output()
        self.layers[-1].deltas[0] = activation * (1 - activation) * (activation - y_actual)

    def get_hidden_delta(self, layer: int, node_index: int) -> float:
        delta: float = 0.0

        current_layer = self.layers[layer]
        next_forward_layer = self.layers[layer+1]
        for k in range(0, next_forward_layer.num_nodes):
            current_node = next_forward_layer.nodes[k]
            delta += current_node.weights[node_index] * next_forward_layer.deltas[k]

        delta = delta * current_layer.activations[node_index] * (1 - current_layer.activations[node_index])
        return delta

    def hidden_back_prop(self, layer: int):
        current_layer: NeuralNetworkLayer = self.layers[layer]

        for i in range(0, current_layer.num_nodes):
            current_layer.deltas[i] = self.get_hidden_delta(layer, i)
        
    def back_prop(self, y_actual: float):
        self.output_back_prop(y_actual)
        for i in range(self.num_layers - 2, 0, -1):
            self.hidden_back_prop(i)

        for i in range(1, len(self.layers)):
            current_layer = self.layers[i]
            prev_layer = self.layers[i-1]
            current_layer.update_layer(prev_layer.activations, self.learning_rate)
            
def avg_squared_error(nn: NeuralNetwork, X: List[List[float]], y: List[float]) -> float:
    error_sum = 0.0
    for xi, yi in zip(X, y):
        output = nn.forward_pass(xi)
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

        forward_output = nn.forward_pass(x_instance)
        nn.back_prop(y_instance)

        train_error = avg_squared_error(nn, train_X, train_y)
        test_error = avg_squared_error(nn, test_X, test_y)

        print(f"In iteration {i+1}:")
        print(f"Forward pass output: {forward_output:.4f}")
        print(f"Average squared error on training set ({num_train} instances): {train_error:.4f}")
        print(f"Average squared error on test set ({num_test} instances): {test_error:.4f}\n")
main()
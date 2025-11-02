'''
Group Members: Lalith Vennapusa, Oscar Ruenes
Date: 10/30/2025
Title: CS 4375 Assignment 4 Neural Networks
Desc: Neural network implementation trained on given learning rate, training/testing data, 
number of iterations, and number of hidden layers and hidden nodes.
'''

import argparse
import numpy as np

from typing import List, Tuple


parser = argparse.ArgumentParser()
parser.add_argument("train_path", type=str, help="File path to training data")
parser.add_argument("test_path", type=str, help="File path to test data")
parser.add_argument("num_hidden_layers", type=int, help="Number of hidden layers")
parser.add_argument("num_hidden_nodes", type=int, help="Number of hidden nodes")
parser.add_argument("learning_rate", type=float, help="Learning rate for training")
parser.add_argument("num_iterations", type=int, help="Number of training iterations")


args = parser.parse_args()

try:
    with open(args.train_path, 'r') as file:
        training_data_lines = [line.strip().split('\t') for line in file.read().splitlines() if line.strip()]
except:
    print(f"Error with train_path occured")
    exit(1)

try:
    with open(args.test_path, 'r') as file:
        test_data_lines = [line.strip().split('\t') for line in file.read().splitlines() if line.strip()]
except:
    print(f"Error with test_path occured")
    exit(1)

class NeuralNetworkNode:
    def __init__(self, num_inputs: int):
        self.num_inputs = num_inputs
        self.weights = np.zeros(num_inputs)
        self.bias = 1.0

    def activate(self, inputs: np.ndarray) -> float:
        z = np.dot(self.weights, inputs) + self.bias
        return self.sigmoid(z)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def update(self, new_weights, new_bias):
        self.weights = new_weights
        self.bias = new_bias
    
class NeuralNetwork:
    def __init__(self, num_of_inputs: int, num_hidden_layers: int, num_hidden_nodes: int, learning_rate: float):
        self.num_of_inputs = num_of_inputs
        self.num_hidden_layers = num_hidden_layers
        self.num_layers = num_hidden_layers + 1
        self.num_hidden_nodes = num_hidden_nodes
        self.learning_rate = learning_rate

        self.layers: List[List[NeuralNetworkNode]] = self.create_layers()
        self.layer_inputs: List[np.ndarray] = [np.zeros(num_of_inputs)] + [np.zeros(num_hidden_nodes) for _ in range(num_hidden_layers)] + [np.zeros(1)]
        print(f"Created a inputs array with shape {self.layer_inputs}")

    def create_layers(self) -> List[List[NeuralNetworkNode]]:
        layers: List[List[NeuralNetworkNode]] = []
        prev_inputs = self.num_of_inputs
        for _ in range(0, self.num_hidden_layers):
            layer: List[NeuralNetworkNode] = []
            for _ in range(0, self.num_hidden_nodes):
                node = NeuralNetworkNode(prev_inputs)
                layer.append(node)

            layers.append(layer)
            prev_inputs = len(layer)

        # Output layer
        layers.append([NeuralNetworkNode(prev_inputs)])
        layers[-1][0].bias = 0

        print(f"Created a nn with {len(layers)} layers with {len(layers[0])} nodes in each hidden layer")
        return layers

    def forward_pass(self, input_data) -> float:
        self.layer_inputs[0] = input_data
        for i in range(0, self.num_layers):
            for j in range(0, len(self.layers[i])):
                nn_node = self.layers[i][j]
                self.layer_inputs[i+1][j] = nn_node.activate(self.layer_inputs[i])

        

        return self.layer_inputs[-1][0]
        
    def back_propagation(self, actual_output: float):
        output_delta = self.output_layer_update(actual_output)

    def output_layer_update(self, actual_output: float) -> float:
        output_layer = self.num_layers
        assert self.layer_inputs[output_layer].shape[0] == 1
        activation = self.layer_inputs[output_layer][0]
        delta = activation * (1 - activation) * (activation - actual_output)

        output_node = self.layers[output_layer - 1][0]

        new_weight: np.ndarray = np.zeros(output_node.num_inputs, dtype=float)
        for i in range(0,output_node.num_inputs):
            new_weight[i] = output_node.weights[i] - self.learning_rate * delta * self.layer_inputs[output_layer-1][i]

        new_bias = output_node.bias - self.learning_rate * delta
        output_node.update(new_weight, new_bias)

        return delta

class NN_Error:
    def __init__(self):
        self.sum = 0
        self.num_of_instances = 0
    
    def add(self, y_hat: float, y_actual: float): 
        self.sum += (y_actual - y_hat)**2
        self.num_of_instances += 1

    def clear(self):
        self.sum = 0
        self.num_of_instances = 0

    def get(self):
        return 0.5 * self.sum * 1 / self.num_of_instances

def main():
    if len(vars(args)) != 6:
        print("Invalid number of arguments, use train_path, test_path, num_hidden_layers, num_hidden_nodes, learning_rate, num_iterations")
        exit(1)

    input_labels = training_data_lines[0][0:-1]
    output_label: str = training_data_lines[0][-1]

    num_of_inputs: int = len(input_labels)

    nn = NeuralNetwork(num_of_inputs, args.num_hidden_layers, args.num_hidden_nodes, args.learning_rate)

    training_instances: List[List[float]] = [[float(x) for x in line[0:-1]] for line in training_data_lines[1:]]
    np_training_instances = np.array(training_instances, dtype=float)
    class_instances: List[float] = [float(line[-1]) for line in training_data_lines[1:]]
    
    print(np_training_instances.shape)
    num_of_instances = np_training_instances.shape[0]

    for _ in range(0, args.num_iterations):
        nn_error: NN_Error = NN_Error()
        for i in range(0, num_of_instances):
            output = nn.forward_pass(np_training_instances[i])
            if i == 0:
                print(output)
            nn_error.add(output, class_instances[i])
            nn.back_propagation(class_instances[i])
        print(f"Average error training set ({num_of_instances} instances): {nn_error.get()}")

    

main()
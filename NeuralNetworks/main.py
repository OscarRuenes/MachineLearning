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
        # How many inputs a node has
        self.num_inputs = num_inputs
        # The weights corresponding to each input
        self.weights = np.zeros(num_inputs)
        # Bias associated with Node
        self.bias = 0

    def activate(self, inputs: np.ndarray) -> float:
        # Simple dot product
        z = np.dot(self.weights, inputs) + 1 * self.bias
        # Activation Function
        return self.sigmoid(z)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def update(self, new_weights, new_bias):
        # "Nudge" the weights accordingly
        self.weights -= new_weights
        self.bias -= new_bias
    
class NeuralNetwork:
    def __init__(self, num_of_inputs: int, num_hidden_layers: int, num_hidden_nodes: int, learning_rate: float):
        # How many inputs does the nn take
        self.num_of_inputs = num_of_inputs
        
        self.num_hidden_layers = num_hidden_layers
        self.num_layers = num_hidden_layers + 1
        # Nodes in each hidden layer
        self.num_hidden_nodes = num_hidden_nodes
        self.learning_rate = learning_rate

        # Layers of nodes that have weights. Input layer is not included because it does not have weights
        self.layers: List[List[NeuralNetworkNode]] = self.create_layers()

        # The activations of the previous iteration of forward pass
        # Includes input layer as well so activations of self.layers[i][j] = self.layer_inputs[i+1][j]
        self.layer_inputs: List[np.ndarray] = [np.zeros(num_of_inputs)] + [np.zeros(num_hidden_nodes) for _ in range(num_hidden_layers)] + [np.zeros(1)]

    def create_layers(self) -> List[List[NeuralNetworkNode]]:
        # Layers is the list of all the NNNodes 
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
        
        # Final shape: num_hidden_layers * (NNNode(num_of_inputs) * num_hidden_nodes) + NNNode(num_hidden_layers)
        return layers

    def forward_pass(self, input_data: np.ndarray) -> float:
        # Storing these activations in layer_inputs as we go
        self.layer_inputs[0] = input_data
        for i in range(0, self.num_layers):
            # Go through each node in a layer and activate it
            for j in range(0, len(self.layers[i])):
                nn_node = self.layers[i][j]
                self.layer_inputs[i+1][j] = nn_node.activate(self.layer_inputs[i])

        # Return output layer
        return self.layer_inputs[-1][0]
        
    def back_propagation(self, actual_output: float):
        # Different algos for output layer and hidden layer
        output_delta = self.output_layer_update(actual_output)
        self.hidden_layer_update(np.array([output_delta]))



    def output_layer_update(self, actual_output: float) -> float:
        # delta_i = (activation_i - actual_i) * (activation_i) * (1 - activation_i) 
        # W_ij (weight from j in previous layer to i in current layer) -= learning_rate * delta_i * activation_j
        # B_i (Bias) -= learning_rate * delta_i

        output_layer = self.num_layers
        
        activation = self.layer_inputs[output_layer][0]
        delta = activation * (1 - activation) * (activation - actual_output)

        output_node = self.layers[output_layer - 1][0]

        # Calcualte Nudge to each weight in the current layer, only one node for our case
        weight_update: np.ndarray = np.zeros(output_node.num_inputs, dtype=float)
        for i in range(0,output_node.num_inputs):
            weight_update[i] = self.learning_rate * delta * self.layer_inputs[output_layer-1][i]
        
        bias_update = self.learning_rate * delta
        
        output_node.update(weight_update, bias_update)

        return delta

    def hidden_layer_update(self, output_layer_deltas: np.ndarray):
        # delta_i = (activation_i) * (1 - activation_i) * sum_k(delta_k * W_ki), where k are the nodes in the next layer
        # W_ij (weight from j in previous layer to i in current layer) -= learning_rate * delta_i * activation_j
        # B_i (Bias) -= learning_rate * delta_i
        next_layer_deltas = output_layer_deltas

        # For each layer, going from the layer closest to output layer to the layer closest to input layer
        for i in range(self.num_hidden_layers - 1, -1, -1):
            # Store this layers deltas for next iteration
            new_deltas = np.zeros(self.num_hidden_nodes, dtype=float)
            
            # Calculate deltas and update weights for each node in a layer
            for j in range(0, self.num_hidden_nodes):
                # The current_node we are looking at
                current_node: NeuralNetworkNode = self.layers[i][j]
                delta: float = 0.0

                # Iterate through all the nodes in the next forward layer that the 
                # current node is connected to, to get delta.
                # The summation in the delta formula
                for k in range(0, next_layer_deltas.shape[0]):
                    prev_delta = next_layer_deltas[k]
                    weight = self.layers[i+1][k].weights[j]
                    delta += prev_delta * weight

                # Remember: activations of self.layers[i][j] = self.layer_inputs[i+1][j]
                activation = self.layer_inputs[i+1][j]
                delta = delta * activation * (1 - activation)

                # Calcualte Nudge to each weight in the current layer's current node
                weight_update: np.ndarray = np.zeros(current_node.num_inputs, dtype=float)
                for l in range(0,current_node.num_inputs):
                    weight_update[l] = self.learning_rate * delta * self.layer_inputs[i][l]
                bias_update = self.learning_rate * delta

                # Update the current node
                current_node.update(weight_update, bias_update)

                # keep track of this delta for next iteration
                new_deltas[j] = delta
            
            # Our new deltas for next iteration
            next_layer_deltas = new_deltas
        return 

# Helpful class to calculate Error     
class NN_Error:
    # Just keeps track of the number of additions and the running sum
    def __init__(self):
        self.sum = 0
        self.num_of_instances = 0
    
    def add(self, y_hat: float, y_actual: float): 
        self.sum += (y_actual - y_hat)**2
        self.num_of_instances += 1

    def clear(self):
        self.sum = 0
        self.num_of_instances = 0

    # Calculates the error when self.get is called
    def get(self):
        return self.sum * 1 / self.num_of_instances

def main():
    if len(vars(args)) != 6:
        print("Invalid number of arguments, use train_path, test_path, num_hidden_layers, num_hidden_nodes, learning_rate, num_iterations")
        exit(1)

    # Labels if we need them
    input_labels = training_data_lines[0][0:-1]
    output_label: str = training_data_lines[0][-1]

    # How many inputs are taken by our nn
    num_of_inputs: int = len(input_labels)

    # Init nn
    nn = NeuralNetwork(num_of_inputs, args.num_hidden_layers, args.num_hidden_nodes, args.learning_rate)

    # Differenciate the inputs and outputs in our training instances and make them np.ndarrays to make use of dot product
    training_instances: List[List[float]] = [[float(x) for x in line[0:-1]] for line in training_data_lines[1:]]
    np_training_instances = np.array(training_instances, dtype=float)
    training_class_instances: List[float] = [float(line[-1]) for line in training_data_lines[1:]]

    # Differenciate the inputs and outputs in our test instances and make them np.ndarrays to make use of dot product
    test_instances: List[List[float]] = [[float(x) for x in line[0:-1]] for line in test_data_lines[1:]]
    np_test_instances = np.array(test_instances, dtype=float)
    test_class_instances: List[float] = [float(line[-1]) for line in test_data_lines[1:]]

    # Number of instances in each class
    num_of_training_instances = np_training_instances.shape[0]
    num_of_test_instances = np_test_instances.shape[0]


    nn_error: NN_Error = NN_Error()
    # Run for num_iterations, learn from one instance per iteration
    for i in range(0, args.num_iterations):
        print(f"At iteration {i+1}:")

        # Re-use instances
        i = i % num_of_training_instances

        # Forward Pass one instance and only learn for that one instance
        output = nn.forward_pass(np_training_instances[i])
        nn.back_propagation(training_class_instances[i])

        print(f"Forward pass output: {output:.4f}")
        
        # Calculate Error after learning from one instace by going over all training and test instances
        nn_error.clear()
        for j in range(0, num_of_training_instances):
            predicted = nn.forward_pass(np_training_instances[j])
            nn_error.add(predicted, training_class_instances[j])

        print(f"Average squared error on training set ({num_of_training_instances} instances): {nn_error.get():.4f}")

        # Clear to make sure that error is only calcuated for instances we are looking at
        nn_error.clear()
        for j in range(0, num_of_test_instances):
            predicted = nn.forward_pass(np_test_instances[j])
            nn_error.add(predicted, test_class_instances[j])

        print(f"Average squared error on test set ({num_of_test_instances} instances): {nn_error.get():.4f}\n")


main()
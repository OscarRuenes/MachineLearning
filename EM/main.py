'''
Author: Lalith Vennapusa, Oscar Ruenes
Date: 11/14/2025
Title: CS 4375 Assignment 5 kNN, MLE, The EM Algorithm, and The Statistical Learning Theory
Desc: This program implements the EM algorithm for Gaussian Mixture Models (GMMs).
'''

import argparse
import math
from typing import List

parser = argparse.ArgumentParser()
parser.add_argument("train_path", type=str)
parser.add_argument("num_gaussians", type=float)
parser.add_argument("num_iterations", type=int)
args = parser.parse_args()

def load_data(path: str):
    try:
        with open(path, 'r') as file:
            lines = [line.strip().split('\t') for line in file if line.strip()]
        data = [float(line[0]) for line in lines]
        return data
    except Exception as e:
        print(f"Error loading data from {path}: {e}")
        exit(1)

def initialize_gaussians(data: List[float], num_gaussians: int):
    data_len = len(data)
    clusters = [[] for _ in range(num_gaussians)]
    for i in range(data_len):
        clusters[(i) % num_gaussians].append(data[i])
    # Placeholder for Gaussian parameters initialization
    means = []
    variances = []
    priors = []
    for cluster in clusters:
        if cluster:
            mean = sum(cluster) / len(cluster)
            variance = sum((x - mean) ** 2 for x in cluster) / len(cluster)
        else:
            mean = 0.0
            variance = 1.0
        means.append(mean)
        variances.append(variance)
    priors = [len(cluster) / data_len for cluster in clusters]
    gaussians = [
        [clusters[i], (means[i], variances[i], priors[i])]
        for i in range(num_gaussians)
    ]
    return gaussians

def e_step():
    # Placeholder for E-step implementation
    pass
def m_step():
    # Placeholder for M-step implementation
    pass

def print_iteration(iteration: int, gaussians: List):
    print(f"After iteration {iteration}:")
    for i, (data_points, (mean, variance, prior)) in enumerate(gaussians):
        print(f"Gaussian {i+1}: mean = {mean:.4f}, variance = {variance:.4f}, prior = {prior:.4f}")
    print()

def main():
    if len(vars(args)) != 3:
        print("Invalid number of arguments, use train_path, num_gaussians, num_iterations")
        exit(1)
    #array of float data points
    data = load_data(args.train_path)
    
    num_gaussians = int(args.num_gaussians)
    num_iterations = args.num_iterations
    gaussians = initialize_gaussians(data, num_gaussians)

    print_iteration(0, gaussians)

    for iteration in range(num_iterations):
        e_step()
        m_step()
        print_iteration(iteration + 1, gaussians)
main()
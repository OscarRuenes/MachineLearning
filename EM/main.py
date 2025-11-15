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

#p(x) formula
def gaussian_pdf(x, mean, variance):
    return (1.0 / math.sqrt(2 * math.pi * variance)) * math.exp(-(x - mean) ** 2 / (2 * variance))


def e_step(gaussians, data):
    num_gaussians = len(gaussians)
    data_len = len(data)

    gamma = [[0.0] * num_gaussians for _ in range(data_len)]

    for n, x in enumerate(data):
        # compute denominator = sum_k π_k * N(x | μ_k, σ_k^2)
        den = 0.0
        numerators = []
        for k in range(num_gaussians):
            #get estimates
            mean, variance, prior = gaussians[k][1]
            #get each numerator
            num = prior * gaussian_pdf(x, mean, variance)
            numerators.append(num)
            #denominator is the sum of all numerators
            den += num
        #list of gammas for each gaussian
        for k in range(num_gaussians):
            gamma[n][k] = numerators[k] / den
    #list of lists of gammas, each sublist corresponds to a data point and each element in the sublist corresponds to a gaussian
    return gamma


def m_step(gaussians, data, gamma):
    data_len = len(data)
    num_gaussians = len(gaussians)
    #update each gaussian's parameters
    for k in range(num_gaussians):
        gamma_sum = sum(gamma[n][k] for n in range(data_len))
        #prior rule
        prior = gamma_sum / data_len
        #mean rule
        mean = sum(gamma[n][k] * data[n] for n in range(data_len)) / gamma_sum
        #variance rule
        variance = sum(gamma[n][k] * (data[n] - mean) ** 2 for n in range(data_len)) / gamma_sum
        gaussians[k][1] = (mean, variance, prior)


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
    gamma = None
    print_iteration(0, gaussians)
    for iteration in range(num_iterations):
        #list of lists of gammas
        gamma = e_step(gaussians, data)
        #update gaussian parameters
        m_step(gaussians, data, gamma)
        print_iteration(iteration + 1, gaussians)
main()
'''
Group Members: Lalith Vennapusa, Oscar Ruenes
Date: 9/22/2025
Title: CS 4375 Assignment 2 Bayesian Learning
Desc: Naive Bayes learning algorithm for binary classification tasks, assumes no missing values for features and binary feature values. Takes data from train.dat and test.dat files, with a header line for the
titles of features/classification and further lines containing the binary values of each feature and classification. Prints training set and test set accuracies.
'''
import argparse

from typing import List, Tuple

parser = argparse.ArgumentParser()
parser.add_argument("train_path", type=str, help="File path to training data")
parser.add_argument("test_path", type=str, help="File path to test data")

args = parser.parse_args()

try:
    with open(args.train_path, 'r') as file:
        training_data_lines = file.read().splitlines()
except:
    print(f"Error with train_path occured")
    exit(1)

try:
    with open(args.test_path, 'r') as file:
        test_data_lines = file.read().splitlines()
except:
    print(f"Error with test_path occured")
    exit(1)

#Input: takes a list of strings, of the form 1\t0\t...\n1\t0... where each line represents a data instance and the whole string represents the data set
#Output: a tuple of tuples, each element tuple containing a data instance
def extract_data(data_str: List[str]) -> Tuple[Tuple[int, ...], ...]:
    num_data_instances = len(data_str)      #number of instances + header
    data_instances = tuple()
    for i in range(1, num_data_instances):  #runs from 1 to number of instances + 1 - 1 = number of instances, skips over header
        if (data_str[i] == ""): 
            continue
        data_instance_str = data_str[i].split()
        

        data_instance = tuple(map(int, data_instance_str))
        data_instances = data_instances + (data_instance,)
    return data_instances


def nothing(x: float, _: float):
    return x

#Input: takes a list of tuples, each tuple being an instance to the given training data containing any number of binary-valued features and one binary-valued classification
#Output: returns a list of lists of tuples, with the first list being the list of probabilities given in tuples; for features: (P(X=0), P(X=1)), and for classification: (0, P(class=0))
#Similarly element 2 of the list contains a list of tuples where the first elements are (P(X=0), P(X=1)) and the last element is (0, P(class=1))
def create_bayes_classifier(training_data):
    #count number of 1s for each feature, given the classification
    frequencies_c1 = list(0 for _ in range(len(training_data[0])))
    frequencies_c0 = list(0 for _ in range(len(training_data[0])))
    num_data_instances = len(training_data)
    for i in range(0, num_data_instances):
        data_instance = training_data[i]
        if data_instance[len(data_instance)-1]:
            for j in range(0,len(data_instance)):
                if data_instance[j]:
                    frequencies_c1[j] = frequencies_c1[j] + 1
        else:
            for j in range(0,len(data_instance)):
                if data_instance[j]:
                    frequencies_c0[j] = frequencies_c0[j] + 1
            #number of 0 classifications, i.e. how many instances belong to class 0
            frequencies_c0[len(frequencies_c0)-1] += 1

    #get probabilities of each feature given their class (simple frequency probabilities)
    probabilities_c1 = list(frequencies_c1)
    for i in range (0,len(probabilities_c1)-1):
        probabilities_c1[i] = probabilities_c1[i] / frequencies_c1[len(frequencies_c1)-1]       #last frequency counts the number of instances within this class

    probabilities_c0 = list(frequencies_c0)
    for i in range (0,len(probabilities_c0)-1):
        probabilities_c0[i] = probabilities_c0[i] / frequencies_c0[len(frequencies_c0)-1]

    #create tuple for (P(X=0), P(X=1)) for each feature X
    bayes_classifier_c1 = [(0, 0) for _ in range(len(probabilities_c1))]
    for i in range(0, len(probabilities_c1)-1):
        bayes_classifier_c1[i] = (nothing(1 - probabilities_c1[i], 2), nothing(probabilities_c1[i], 2))
    bayes_classifier_c0 = [(0, 0) for _ in range(len(probabilities_c0))]
    for i in range(0, len(probabilities_c0)-1):
        bayes_classifier_c0[i] = (nothing(1 - probabilities_c0[i], 2), nothing(probabilities_c0[i], 2))
    
    #Calculate P(class=0) P(class=1)
    bayes_classifier_c1[len(probabilities_c1)-1] = (0, nothing(frequencies_c1[len(frequencies_c1)-1] / len(training_data), 2))
    bayes_classifier_c0[len(probabilities_c0)-1] = (0, nothing(frequencies_c0[len(frequencies_c0)-1] / len(training_data), 2))
    #Returns list of 
    return [bayes_classifier_c0, bayes_classifier_c1]

#Input: a bayes classifier from before (list of two lists containing ordered pairs of (P(X=0), P(X=1)) or (0,P(class=0/1))), and a list of labels given, of size n with n-1 feature titles and the final being the class title
#Output: nothing returned, printing of bayes probabilities to stdout
def print_bayes_classifier(bayes_classifier, labels):
    #for each class
    for i in range(0, len(bayes_classifier)):
        #print class probability
        print(f"P(" + str(labels[len(labels)-1]) + "=" + str(i) + ")=" + f"{bayes_classifier[i][len(bayes_classifier[i])-1][1]:.2f}" + " ", end="")
        #print conditional probabilities given class i
        for j in range(0, len(bayes_classifier[i])-1):
            #each feature has k possible values
            for k in range(0, len(bayes_classifier[i][j])):
                print(f"P(" + str(labels[j]) + "=" + str(k) + "|" + str(i) + ")=" + f"{bayes_classifier[i][j][k]:.2f}", end=" ")
        #newline between classes
        print()


def evaluate_classifier(bayes_classifier: List[List[Tuple[int, int]]], data_instances: Tuple[Tuple[int, ...], ...]) -> float:
    # tracking the total data_instances and the number of correct predictions for final 
    # accuracy calculation
    num_of_data_instances = len(data_instances)
    correct_predictions = 0

    num_of_features = len(data_instances[0])
    
    # for each data instance in the list of data we got
    for data_instance in data_instances:
        # measures the probability that the given data_instance with the given features
        # is of the correct class vs the opposite class
        for_probability = 1.0
        against_probability = 1.0

        data_instance_class = data_instance[num_of_features-1]

        # NAIVE BAYES ASSUMPTION
        for i in range(num_of_features-1):
            # probability *= probability that the current feature is equal to the given feature given the class
            for_probability *= bayes_classifier[data_instance_class][i][data_instance[i]]
            against_probability *= bayes_classifier[1-data_instance_class][i][data_instance[i]]
        # multiply the probability of a data instance being the class
        for_probability *= bayes_classifier[data_instance_class][num_of_features-1][1]
        against_probability *= bayes_classifier[1-data_instance_class][num_of_features-1][1]

        if (for_probability > against_probability):
            correct_predictions += 1
    
    return (correct_predictions / num_of_data_instances)*100

def main():
    #Get labels and data
    labels = training_data_lines[0].split()
    training_data = extract_data(training_data_lines)
    test_data = extract_data(test_data_lines)
    

    bayes_classifier = create_bayes_classifier(training_data)
    print_bayes_classifier(bayes_classifier, labels)

    training_accuracy = evaluate_classifier(bayes_classifier, training_data)
    print(f"\nAccuracy on training set ({len(training_data)} instances): " + f"{training_accuracy:.2f}%")

    test_accuracy = evaluate_classifier(bayes_classifier, test_data)
    print(f"\nAccuracy on test set ({len(test_data)} instances): " + f"{test_accuracy:.2f}%")

main()
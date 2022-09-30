# HW#4 Machine Learning 10-601, Meiirbek Islamov
# Logistic Regression

# import the necessary libraries
import sys
import numpy as np

args = sys.argv
assert(len(args) == 9)
formatted_train_input = args[1] # Path to the training data .tsv file
formatted_valid_input = args[2] # Path to the validation input .tsv file
formatted_test_input = args[3] # Path to the test input .tsv file
dict_input = args[4] # Path to the dictionary input .txt
train_out = args[5] # Path to output .tsv file to which the feature extractions on the train- ing data should be written
test_out = args[6] # Path to output .tsv file to which the feature extractions on the validation data should be written
metrics_out = args[7] # Path to output .tsv file to which the feature extractions on the test data should be written
num_epoch = int(args[8]) # Model 1 feature set or the Model 2 feature set

# Functions
# Read formatted data
def read_formatted(input):
    with open(input, 'r') as f_in:
        lines = f_in.readlines()
    data = np.array([list(l.split()[1:]) for l in lines], dtype="object")
    label = np.array([int(l.split()[0]) for l in lines])
    data_dict = []
    for i, item in enumerate(data):
        dictionary = {}
        for j, elem in enumerate(item):
            l = elem.split(":")
            dictionary[int(l[0])] = int(l[1])
        data_dict.append(dictionary)
    return data_dict, label

# Read dictionary input
def read_dict(input):
    with open(input, 'r') as f_in:
        lines = f_in.readlines()
    data = np.array([list(l.split()) for l in lines])
    return data


# Single SGD step
def sgd_single(intercept, params, learning_rate, features, label, N):
    product = 0
    for key, value in features.items():
        product += params[key]
    product_int = intercept + product
    sigma = np.exp(product_int)/(1 + np.exp(product_int))
    diff = sigma - label
    # Update the model parameters and the intercept separately
    for key, value in features.items():
        params[key] -= (learning_rate / N) * diff
    intercept -= (learning_rate / N) * diff
    return intercept, params


# Multiple SGD steps
def sgd_many(intercept, params, label, train_data, num_epoch, learn_rate):
    for i in range(num_epoch):
        for j, item in enumerate(label):
            intercept, params = sgd_single(intercept, params, learn_rate, train_data[j], label[j], len(train_data))
    return intercept, params


# Predict labels
def predict_labels(intercept, params, features):
    label = []
    for i, instance in enumerate(features):
        product = 0
        for key, value in instance.items():
            product += params[key]
        product_int = intercept + product
        sigma = np.exp(product_int)/(1 + np.exp(product_int))
        if sigma > 0.5:
            y = 1
        else:
            y = 0
        label.append(y)
    return label


# Calculate errors
def calculate_error(label_true, label_predicted):
    n = 0
    for i, item in enumerate(label_true):
        if item != label_predicted[i]:
            n += 1
    error = n/len(label_true)
    return error


# Write labels
def write_labels(predicted_label, filename):
    with open(filename, 'w') as f_out:
        for label in predicted_label:
            f_out.write(str(label) + '\n')


# Write errors
def write_error(train_error, test_error, filename):
    with open(filename, 'w') as f_out:
        f_out.write("error(train): " + str(train_error) + "\n")
        f_out.write("error(test): " + str(test_error) + "\n")


# Main body
# Read the inputs
train_data, label_train = read_formatted(formatted_train_input)
valid_data, label_valid = read_formatted(formatted_valid_input)
test_data, label_test = read_formatted(formatted_test_input)
dict_txt = read_dict(dict_input)
words = list(dict_txt[:, 0])
volume = len(dict_txt)

# Learning rate
learn_rate = 0.1
# Initialise the model parameters
params = [0] * volume
intercept = 0

# Learn the logistic function model parameters using the Stochastic Gradient Decsent (SGD)
intercept, params = sgd_many(intercept, params, label_train, train_data, num_epoch, learn_rate)

# Predict the labels
predicted_label_train = predict_labels(intercept, params, train_data)
predicted_label_test = predict_labels(intercept, params, test_data)

# Write labels
write_labels(predicted_label_train, train_out)
write_labels(predicted_label_test, test_out)

# Calculate errors
error_train = calculate_error(label_train, predicted_label_train)
error_test = calculate_error(label_test, predicted_label_test)

# Write errors
write_error(error_train, error_test, metrics_out)

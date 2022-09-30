# HW#4 Machine Learning 10-601, Meiirbek Islamov
# Logistic Regression

# import the necessary libraries
import sys
import numpy as np

args = sys.argv
assert(len(args) == 9)
train_input = args[1] # Path to the training data .tsv file
validation_input = args[2] # Path to the validation input .tsv file
test_input = args[3] # Path to the test input .tsv file
dict_input = args[4] # Path to the dictionary input .txt
formatted_train_out = args[5] # Path to output .tsv file to which the feature extractions on the train- ing data should be written
formatted_validation_out = args[6] # Path to output .tsv file to which the feature extractions on the validation data should be written
formatted_test_out = args[7] # Path to output .tsv file to which the feature extractions on the test data should be written
feature_flag = int(args[8]) # Model 1 feature set or the Model 2 feature set

# Functions
def read_data(input):
    with open(input, 'r') as f_in:
        lines = f_in.readlines()
    data = np.array([list(l.split()[1:]) for l in lines], dtype="object")
    label = np.array([l.split()[0] for l in lines])
    return data, label

def read_dict(input):
    with open(input, 'r') as f_in:
        lines = f_in.readlines()
    data = np.array([list(l.split()) for l in lines])
    return data

# Model 1
def model_1(train_data, words):
    sparse_repr = []
    for i, item in enumerate(train_data):
        dict = {}
        for elem in item:
            if elem in words:
                dict[words.index(elem)] = 1
        sparse_repr.append(dict)
    return sparse_repr

# Model 2
def model_2(train_data, words):
    sparse_repr = []
    for i, item in enumerate(train_data):
        dict = {}
        values, counts = np.unique(item, return_counts=True)
        val_list = list(values)
        for elem in item:
            if elem in words and counts[val_list.index(elem)] < 4:
                dict[words.index(elem)] = 1
        sparse_repr.append(dict)
    return sparse_repr

# write
def write_formatted_feature(label, sparse_repr, filename):
    with open(filename, 'w') as f_out:
        for i, item in enumerate(label):
            feature = "\t".join(str(key) + ":" + str(value) for key, value in sparse_repr[i].items())
            f_out.write(str(item) + "\t" + str(feature) +"\n")

# Training
train_data, train_label = read_data(train_input)
validation_data, validation_label = read_data(validation_input)
test_data, test_label = read_data(test_input)
dict_txt = read_dict(dict_input)
words = list(dict_txt[:, 0])

if feature_flag == 1:
    train_model = model_1(train_data, words)
    validation_model = model_1(validation_data, words)
    test_model = model_1(test_data, words)
elif feature_flag == 2:
    train_model = model_2(train_data, words)
    validation_model = model_2(validation_data, words)
    test_model = model_2(test_data, words)

write_formatted_feature(train_label, train_model, formatted_train_out)
write_formatted_feature(validation_label, validation_model, formatted_validation_out)
write_formatted_feature(test_label, test_model, formatted_test_out)

'''
Name:   Balaputradewa Ratuwina
ID:     1001761950
'''

import numpy as np

# function to find standard deviation w/denominator as N - 1
# if < 0.01, return 0.01
def std(data_column, avg):
    N = len(data_column)
    # avoid division by 0
    if N <= 1:
        return 0.01
    d2 = abs(data_column - avg) ** 2
    var = np.sum(d2) / (N - 1)
    std = np.sqrt(var)
    if std < 0.01:
        return 0.01
    else:
        return std

def gaussian(x, mean, std):
    if std == 0:
        return 0.0
    return np.exp( - (x - mean) ** 2 / (2 * std ** 2)) / (np.sqrt(2 * np.pi) * std)

def naive_bayes(training_file, test_file):
    # read from the training files
    training_data = np.loadtxt(training_file)
    test_data = np.loadtxt(test_file)

    # for each class, for each attribute, get gaussian and compute p_C
    rows, cols = training_data.shape
    dimensions = cols - 1
    classes = int(np.amax(training_data[:, -1])) + 1
    means = np.zeros((classes, dimensions))
    stds = np.zeros((classes, dimensions))
    p_C = np.zeros(classes)

    '''
    Training Stage: For every attribute A, for every class C:
        1. Identify all training examples that belong to class C
        2. For all training examples selected, find:
            a. Mean
            b. Standard Deviation
        3. Use values to get the Gaussian for the density P(A|C) of attribute A given class C
        4. Compute the prior p(C) simply as percentage of training examples whose class label is C
    '''

    # find mean and std of each feature in each class
    # store as 2D array [[mean, std], [mean, std], ...] with index being index - 1
    for i in range(1, classes):
        class_data = training_data[training_data[:, -1] == i]

        # assign default value/skip calculations
        if len(class_data) == 0:
            means[i, :] = 0.0
            stds[i, :] = 0.01
            p_C[i] = 0.0
            continue

        class_data_features = np.delete(class_data, -1, 1)
        for j in range(dimensions):
            mean = np.mean(class_data_features[:, j])
            std_dev = std(class_data_features[:, j], mean)
            means[i][j] = mean
            stds[i][j] = std_dev
            print("Class %d, attribute %d, mean = %.2f, std = %.2f" % (i, j + 1, mean, std_dev))

        # compute P(C_j given X) using Bayes rule
        p_C[i] = len(class_data) / rows
    
    '''
    Test Stage: For each test input X = (X_1, ..., X_d):
                For each class C_j in C_1, ..., C_k
                    For each attribute value X_i in X_1, ..., X_d:
                        Compute p(X_i given C_j) using Gaussian density
    '''
    rows_test, cols_test = test_data.shape
    total_accuracy = 0

    for i in range(rows_test):
        x = test_data[i, :-1]
        true_class = int(test_data[i, -1])
        probabilities = np.zeros(classes)

        for c in range(classes):
            p_x_given_C = 1.0
            for j in range(dimensions):
                p_x_given_C *= gaussian(x[j], means[c, j], stds[c, j])
            probabilities[c] = p_x_given_C * p_C[c]
    
        # Normalize
        sum_probs = np.sum(probabilities)
        if sum_probs > 0:
            probabilities /= sum_probs
        
        predicted_class = np.argmax(probabilities)
        predicted_probability = np.max(probabilities)

        # Calculate accuracy
        max_prob = np.max(probabilities)
        tied_classes = np.where(probabilities == max_prob)[0]
        accuracy = 0

        if true_class in tied_classes:
            accuracy = 1.0 / len(tied_classes)
        else:
            accuracy = 0.0
        
        total_accuracy += accuracy

        print("ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f" % (i + 1, predicted_class, predicted_probability, true_class, accuracy))

    overall_accuracy = total_accuracy / rows_test
    print("classification accuracy=%6.4f" % overall_accuracy)
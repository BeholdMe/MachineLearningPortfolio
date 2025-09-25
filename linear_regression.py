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

def create_polymonial_features(X, degree):
    '''
        Create polymonial feature up to specific degree
        If degree == 1 -> φ(x) = (1, x1, x2, ..., xD)T
        If degree == 2 -> φ(x) = (1, x1, (x1)2, x2, (x2)2 ..., xD, (xD)2)T
        If degree == 3 -> φ(x) = (1, x1, (x1)2, (x1)3, x2, (x2)2, (x2)3, ..., xD, (xD)2, (xD)3)T
    '''
    n_samples, n_features = X.shape
    phi = np.ones((n_samples, 1))

    # add for OG feature
    for i in range(n_features):
        for d in range(1, degree + 1):
            phi = np.hstack((phi, (X[:, i:i+1] ** d)))
    
    return phi

def linear_regression(training_file, test_file, degree, lambda1):
    # load training data and test data
    training_data = np.loadtxt(training_file)
    test_data = np.loadtxt(test_file)

    # Get all columns except last from training and test data (features)
    x_train = training_data[:, :-1]
    y_train = training_data[:, -1]
    x_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    # Create polymonial feature from training data
    phi_train = create_polymonial_features(x_train, degree)

    # calculate weights using regularized least squares
    n_features = phi_train.shape[1]
    I = np.eye(n_features)
    # Don't regularize bias term w0
    if n_features > 0:
        I[0, 0] = 0
    phi_T = phi_train.T
    A = np.dot(phi_T, phi_train) + lambda1 * I
    b = np.dot(phi_T, y_train)
    w = np.dot(np.linalg.pinv(A), b)

    # Print weightss
    for i, weight in enumerate(w):
        print(f"w{i}={weight:.4f}")
    
    print()

    # create poly feats for each test data
    phi_test = create_polymonial_features(x_test, degree)

    # calc the predicitons and errors for each test data
    predictions = np.dot(phi_test, w)
    errors = (predictions - y_test) ** 2

    # print test results
    for i in range(len(predictions)):
        print(f"ID={i+1:5d}, output={predictions[i]:14.4f}, target value = {y_test[i]:10.4f}, squared error = {errors[i]:.4f}")

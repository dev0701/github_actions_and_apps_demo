import numpy as np
from numpy import genfromtxt
import random
import math
# Input:    name of csv
#           target district to be used as label y = +1
#           the number of samples in the training dataset (n)
# Output:   numpy vector X_train, of n rows, d columns
#           numpy vector y_train, of n rows, 1 column
#           numpy vector X_val
#           numpy vector y_val
#           numpy vector X_test
#           numpy vector y_test
def run(csv, target_district, neg_district, n):
    raw_array = genfromtxt(csv, delimiter = ',')
    raw_size = raw_array.shape[0]
    d_plus_1 = raw_array.shape[1]
    d = d_plus_1 - 1
    X_train = np.zeros((n, d))
    y_train = np.zeros((n, 1))
    i = 0
    while i < n:
        r = random.randrange(raw_size)
        if i % 2 == 0:
            if raw_array[r][0] == target_district:
                y_train[i][0] = 1
                for t in range(d):
                    X_train[i][t] = raw_array[r][t + 1]
                i = i + 1
                raw_array = np.delete(raw_array, r, 0)
                raw_size = raw_size - 1
        else:
            if raw_array[r][0] == neg_district:
                y_train[i][0] = -1
                for t in range(d):
                    X_train[i][t] = raw_array[r][t + 1]
                i = i + 1
                raw_array = np.delete(raw_array, r, 0)
                raw_size = raw_size - 1

    X_test = np.zeros((math.floor(n * (1 / 7)), d))
    y_test = np.zeros((math.floor(n * (1 / 7)), 1))
    n_test = X_test.shape[0]
    i = 0
    while i < n_test:
        r = random.randrange(raw_size)
        if i % 2 == 0:
            if raw_array[r][0] == target_district:
                y_test[i][0] = 1
                for t in range(d):
                    X_test[i][t] = raw_array[r][t + 1]
                i = i + 1
                raw_array = np.delete(raw_array, r, 0)
                raw_size = raw_size - 1
        else:
            if raw_array[r][0] == neg_district:
                y_test[i][0] = -1
                for t in range(d):
                    X_test[i][t] = raw_array[r][t + 1]
                i = i + 1
                raw_array = np.delete(raw_array, r, 0)
                raw_size = raw_size - 1

    X_val = np.zeros((math.floor(n * (2 / 7)), d))
    y_val = np.zeros((math.floor(n * (2 / 7)), 1))
    n_val = X_val.shape[0]
    i = 0
    while i < n_val:
        r = random.randrange(raw_size)
        if i % 2 == 0:
            if raw_array[r][0] == target_district:
                y_val[i][0] = 1
                for t in range(d):
                    X_val[i][t] = raw_array[r][t + 1]
                i = i + 1
                raw_array = np.delete(raw_array, r, 0)
                raw_size = raw_size - 1
        else:
            if raw_array[r][0] == neg_district:
                y_val[i][0] = -1
                for t in range(d):
                    X_val[i][t] = raw_array[r][t + 1]
                i = i + 1
                raw_array = np.delete(raw_array, r, 0)
                raw_size = raw_size - 1


    return X_train, y_train, X_test, y_test, X_val, y_val
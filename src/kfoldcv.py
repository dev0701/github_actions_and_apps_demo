import numpy as np
from numpy import genfromtxt
import random
import math
import perceptron
import predict
import KNN_new
# Input:    name of csv
#           target district to be used as label y = +1
#           the number of samples in the training dataset (n)
# Output:   numpy vector X_train, of n rows, d columns
#           numpy vector y_train, of n rows, 1 column
#           numpy vector X_val
#           numpy vector y_val
#           numpy vector X_test
#           numpy vector y_test
def run(csv, target_district, neg_district, k, algorithm, k_forKNN):
    raw_array = genfromtxt(csv, delimiter = ',')
    raw_size = raw_array.shape[0]
    d_plus_1 = raw_array.shape[1]
    d = d_plus_1 - 1
    use_train_X = np.zeros((98, d))
    use_train_y = np.zeros((98, 1))
    n = 98

    p = 0
    while p < 98:
        r = random.randrange(raw_size)
        if p % 2 == 0:
            if raw_array[r][0] == target_district:
                use_train_y[p][0] = 1
                for t in range(d):
                    use_train_X[p][t] = raw_array[r][t + 1]
                p = p + 1
                raw_array = np.delete(raw_array, r, 0)
                raw_size = raw_size - 1
        else:
            if raw_array[r][0] == neg_district:
                use_train_y[p][0] = -1
                for t in range(d):
                    use_train_X[p][t] = raw_array[r][t + 1]
                p = p + 1
                raw_array = np.delete(raw_array, r, 0)
                raw_size = raw_size - 1
    
    k_fold_size = math.floor(98/k)
    backup_X = use_train_X
    backup_y = use_train_y

    total_accuracy = 0
    for l in range(k):
        use_train_X = backup_X
        use_train_y = backup_y
        X_train = np.zeros((n - k_fold_size, d))
        y_train = np.zeros((n - k_fold_size, 1))
        X_test = np.zeros((k_fold_size, d))
        y_test = np.zeros((k_fold_size, 1))
        for p in range(k_fold_size):
            y_test[p] = use_train_y[(l * k_fold_size) + p]
            X_test[p] = use_train_X[(l * k_fold_size) + p]
        for p in range(k_fold_size):
            use_train_X = np.delete(use_train_X, l * k_fold_size, 0)
            use_train_y = np.delete(use_train_y, l * k_fold_size, 0)
        for p in range(n - k_fold_size):
            X_train[p] = use_train_X[p]
            y_train[p] = use_train_y[p]
        if(algorithm == 0):     #Perceptron
            perceptron_correct = 0
            theta, iter = perceptron.run(200, X_train, y_train)
            # perceptron testing
            for i in range(len(y_test)):
                print("Prediction: " + str(predict.run(theta, X_test[i])))
                print("Actual: " + str(y_test[i]))
                # print("Theta: " + str(theta))
                if predict.run(theta, X_test[i]) == y_test[i] :
                    perceptron_correct = perceptron_correct + 1
            total_accuracy = total_accuracy + (perceptron_correct/len(y_test))
        if(algorithm != 0):     #KNN
            total_accuracy = total_accuracy + KNN_new.run(k_forKNN, X_train, y_train,
                    X_test, y_test, [], [])[0]
    average_accuracy  = total_accuracy/k

    if algorithm == 0: 
        print("Average perceptron accuracy via k-fold cross validation: \n")
        print("Number of folds (K): " + str(k) + "\n")
        print("Accuracy: " + str(average_accuracy * 100 ) + "%")
    else: 
        print("Average K-nearest neighbors accuracy via k-fold cross validation: \n")
        print("Number of folds (K): " + str(k) + "\n")
        print("Accuracy: " + str(average_accuracy * 100 ) + "%")


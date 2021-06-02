from sklearn.neighbors import KNeighborsClassifier
import numpy as np
def run(k, X_train, y_train, X_test, y_test, X_val, y_val):

    n,d = np.shape(X_train)

    y_train = np.reshape(y_train, (n))


    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    trainAcc = neigh.score(X_test, y_test)
    valAcc = 0
    if(len(X_val) != 0):
        valAcc = neigh.score(X_val, y_val)

    return trainAcc, valAcc

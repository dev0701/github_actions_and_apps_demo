import numpy as np
import numpy.linalg as la

# Input: numpy matrix X of features, with n rows (samples), d columns (features)
#            X[i,j] is the j-th feature of the i-th sample
#        numpy vector y of labels, with n rows (samples), 1 column
#            y[i] is the label (+1 or -1) of the i-th sample
#        numpy vector z of d rows, 1 column
# Output: label (+1 or -1)

def run(X,y,z):
    c = 0
    b = la.norm(np.transpose(z) - X[0])
    ##print(b)

    for t in range(len(y)):
        if la.norm(np.transpose(z) - X[t]) < b:
            c = t
            b = la.norm(np.transpose(z) - X[t])

    label = y[c, 0]

    return label
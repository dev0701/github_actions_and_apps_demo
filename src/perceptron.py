import numpy as np

# Input: maximum number of iterations L
#        numpy matrix X of features, with n rows (samples), d columns (features)
#            X[i,j] is the j-th feature of the i-th sample
#        numpy vector y of labels, with n rows (samples), 1 column
#            y[i] is the label (+1 or -1) of the i-th sample
# Output: numpy vector theta of d rows, 1 column
#        number of iterations that were actually executed (iter+1)

def run(L,X,y):
    theta = np.zeros((len(X[0]), 1))
    for iter in range(L):
        all_points_classified_correctly = True
        for t in range(len(y)):
            dot = 0
            for i in range(len(theta)):
                dot += theta[i, 0]*X[t, i]
            if y[t]*dot <= 0:
                # theta = theta + y[t]*X[t]
                for i in range(len(theta)):
                    theta[i] += y[t]*X[t, i]
                all_points_classified_correctly = False
        if all_points_classified_correctly:
            break
        #print("Theta: " + str(theta))

    return theta, iter+1
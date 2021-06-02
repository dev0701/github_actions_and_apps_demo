import numpy as np

# Input: numpy vector theta of d rows, 1 column
#        numpy vector x of d rows, 1 column
# Output: label (+1 or -1)

def run(theta,x):
    dot = 0
    for i in range(len(theta)):
        dot += theta[i, 0]*x[i]

    if dot > 0:
        label = 1
    else:
        label = -1

    return label
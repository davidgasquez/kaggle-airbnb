
# coding: utf-8

# In[1]:

import numpy as np


# In[175]:

def triangularize(v):
    # Length formula?
    l = (len(v) / 2) + 1 if len(v) % 2 == 0 else (v / 2)

    # Make empty matrix
    matrix = np.zeros((l, l))

    # Fill upper triangle with v
    matrix[np.triu_indices(l, 1)] = v
    # matrix[np.tril_indices(l, -1)] = v

    for i in range(l):
        for j in range(i, l):
            matrix[j][i] = 1 - matrix[i][j]

    np.fill_diagonal(matrix, 0)

    return matrix


# In[172]:

m = np.matrix(
    [
        [0, 1, 0.9, 0.4],
        [0, 0, 0.3, 0],
        [0.1, 0.7, 0, 0.2],
        [0.6, 1, 0.8, 0],
    ]
)

v = [1., 0.9, 0.4, 0.3, 0.,  0.2]


# In[173]:

print triangularize(v)

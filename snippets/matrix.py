
# coding: utf-8

# In[1]:

import numpy as np


# In[175]:

def triangularize(v):
    # Length formula?
    l = (len(v) / 2) + 1 if len(v) % 2 == 0 else (v / 2)

    # Make empty matrix
    tri = np.zeros((l, l))

    # Fill upper triangle with v
    tri[np.triu_indices(l, 1)]  = v
    tri[np.tril_indices(l, -1)]  = v

    # Invert lower triangle
    il = np.tril_indices(l, 0)
    tri[il] = 1 - tri[il]

    return tri


# In[172]:

m = np.matrix(
    [
        [0, 1 ,0.9, 0.4],
        [0, 0 ,0.3, 0],
        [0.1, 0.7 ,0, 0.2],
        [0.6, 1 ,0.8, 0],
    ])

v = [ 1., 0.9, 0.4, 0.3, 0.,  0.2]


# In[173]:

print triangularize(v)

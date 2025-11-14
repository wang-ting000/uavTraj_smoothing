import numpy as np


def add_err(qs,CEP):
    res = qs.copy()
    sigma = CEP/0.6745
    for i in range(len(qs)):
        res[i][0] += np.random.normal(0,sigma,1)
        res[i][1] += np.random.normal(0,sigma,1)
        res[i][2] += np.random.normal(0,sigma,1)
    return res


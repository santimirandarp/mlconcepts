""" 
'compare' the cost and performance of regression and 
multiclassification 
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from functions import *
from csvfns import *

"""
To run we need:
    X: features
    Y: labels
"""

# compare distrib to naive estimation
def results_distribution(Ap):
    """
    Y is the reference set, a pandas column
    Yp are the numpy predictions
    """
    l = Ap.shape[0]
    distribution = (pd.DataFrame(Ap).value_counts()/l)*100
    return distribution

# bring data in
csv = pd.read_csv('winequality-white.csv', delimiter=';')


#create normalized datasets
X,Y,X_test,Y_test = datasets(csv, normalize=True, sliceAt=np.sqrt) 

# convert labels into binary arrays
Y = zero_one(Y)

# run model
W,B,Ap = model(X,Y,10000,lr=2.0)

# print naive model (44-45% accuracy)
print("naive estimation[0], class distribution:\n", naive_estim(csv["quality"] , method="classification"))

# print our own model results
r = pd.DataFrame(Ap.argmax(axis=0))
print("results", results_distribution(r))

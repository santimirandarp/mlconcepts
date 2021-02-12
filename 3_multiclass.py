# multiclass multivariate problem
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

# bring data in
csv = pd.read_csv('winequality-white.csv', delimiter=';')

# numpy test and train datasets
X,Y,X_test,Y_test = datasets(csv, normalize=True, sliceAt=np.sqrt) 

# converts labels to 0/1 arrays
Y = zero_one(Y) 

# run the model
W,B = model(X,Y,5000)



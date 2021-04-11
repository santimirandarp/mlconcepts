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

"Bring data in"
wine = pd.read_csv('winequality-white.csv', delimiter=';')

"Prepare data"
X,Y = toXY(wine, Ykey=False, Yindex=-1)

X = normalize(X)

" splits to create test and train dataframes"
X,Y,X_test,Y_test = trainAndTest(X,Y,sliceAt = np.sqrt)

" convert dataframes to numpy "
X,Y,X_test,Y_test = to_numpy(X,Y,X_test,Y_test) 

"traspose, reshape"
X = X.T # (features, samples)
X_test = X_test.T
Y = Y.reshape(1, Y.shape[0]) # (1 x samples)
Y_test = Y_test.reshape(1, Y_test.shape[0])

"hotenconde the labels for multiclassification"
Y = hotencode(Y) # converts labels to 0/1 arrays

" shape check"
print("X train and test shape", X.shape, X_test.shape)
print("Y train and test shape", Y.shape, Y_test.shape) 
print("original shape was", wine.T.shape)


"Write the model using prev functions"
W,B = fit_generic(X,Y,5000,sigmoid,lr=0.1)


" compare regression and multiclassification "
"this isn't finish, but it calculates rmse and accuracy for 
linear and mulciclass respectively"
"we still need a way to compare the performance"

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
wine = pd.read_csv('winequality-white.csv', delimiter=';')

X,Y = toXY(wine, Ykey=False, Yindex=-1)
X = normalize(X,axis=0)

" splits to create test and train dataframes "
X,Y,X_test,Y_test = trainAndTest(X,Y,sliceAt = np.sqrt)
" convert dataframes to numpy "
X,Y,X_test,Y_test = to_numpy(X,Y,X_test,Y_test) 

"traspose, reshape"
X = X.T # (features, samples)
X_test = X_test.T
Y = Y.reshape(1, Y.shape[0]) # (1 x samples)
Y_test = Y_test.reshape(1, Y_test.shape[0])

# linear calculation
print(X.shape, Y.shape)
#W,B = fit_generic(X,Y,10000,activation=identity,lr=0.1)
#print("results regression")
#print(naive_estim(Y,method="regression"))
#print(metric(X,Y,W,B,method="regression"))

# RESULTS 0.88, 0.75 


"MULTICLASSIFICATION"

"hotencode the labels for multiclassification"
#Y = hotencode(Y) # converts labels to 0/1 arrays
#
#W,B = fit_generic(X,Y,4000,lr=2.0)
#Ap = predict(W,B,X,sigmoid)
#
## results distribution, axis=0 is a line over columns
#Ap = Ap.argmax(axis=0)
#Y = Y.argmax(axis=0)
#
#print("results classification")
#l = Ap.shape[0]
#Ap = pd.DataFrame(Ap).value_counts()*100/l
#Y = pd.DataFrame(Y).value_counts()*100/l
#
#print("predicted distrib\n", Ap, "original distrib\n", Y)

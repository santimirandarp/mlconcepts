# multiclass
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
#Y=Y.reshape(1, Y.shape[0])
#print(Y.shape)
#W,B = model(X,Y,5000, activation=linear)
Y = zero_one(Y) # converts labels to 0/1 arrays
W,B = model(X,Y,5000)



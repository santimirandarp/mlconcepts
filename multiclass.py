# multiclass
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from functions import *

"""
To run we need:
    X: features
    Y: labels
"""

# bring data in
csv = pd.read_csv('winequality-white.csv', delimiter=';')

# generate normalized test and train datasets
X,Y,X_test,Y_test = datasets(csv)
X,Y,X_test, Y_test = to_numpy(X,Y,X_test,Y_test)
Y = zero_one(Y) # converts labels to 0/1 arrays

# forward propagation
def predict(W,B,X,fn):
    """
    W matrix/vector of weights
    B vector/number of biases
    X features
    fn function applied to the linear transform

    Returns the predictions (matrix or vector)
    """
    Z = np.dot(W,X) + B
    return fn(Z)

def cost(X, A, Ap):
    m = X.shape[1]
    cost = -1/m*(np.sum(A*np.log(Ap) + (1-A)*np.log(1-Ap), axis=1, keepdims=True))
    print(cost)
    return cost

# backpropagation
def update(W,B,X,Y,Ap):
    diff = (Y-Ap).T
    W = W + np.dot(X,diff).T*0.001
    B = B + diff.T*0.001
    return W, B

def model(X,Y,it=100,nodes=10):
    # W,B for the 1st calculation
    W,B = initialize(nodes, X.shape[0]) 
    for i in range(it):
      Ap = predict(W,B,X,sigmoid)
      W,B = update(W,B,X,Y,Ap)
      # very sad plot
      if i%(it/10)==0:
          plt.scatter(cost(X,Y,Ap)[0], i)
    plt.show()
    return W,B

W,B = model(X,Y,50000)



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

# normalized, split datasets
X,Y,c,d = datasets(csv)
X,Y,c,d = to_numpy(X,Y,c,d)
Y = zero_one(Y)

# forward propagation
def sigmoid(z):
    """ takes linear piece
    returns the prediction"""
    return 1/(1+np.exp(-z))

def initialize(nodes,features):
    """
    the rows of W are nodes
    the columns of W are features
    the rows of B are nodes

    returns W, B
    """
    w = np.random.rand(nodes, features)
    #each node computes for a set of features
    b = np.random.rand(nodes, 1)
    return w, b

def predict(W,B,X,fn):
    """
    W matrix/vector of weights
    B vector/number of biases
    X features
    fn function applied to the linear transform

    Returns the predictions (matrix or vector)
    """
    A = np.dot(W,X) + B
    return fn(A)

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



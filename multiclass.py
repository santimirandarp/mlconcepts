# shallow network
import numpy as np 
import matplotlib.pyplot as plt

"""
To run we need:
    X: features
    Y: labels
"""

# forward propagation
def sigmoid(z):
    # vector of predictions
    return 1/(1+np.exp(-z))

def initialize(nodes,features):
    """
    the rows of W are nodes
    the columns of W are features
    the rows of B are nodes

    returns W, B
    """
    w = np.zeros((nodes, features))
    #each node computes for a set of features
    b = np.zeros((nodes, 1))
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

def fp(X, nodes):
    """
    X: features
    nodes: nodes
    returns prediction
    """
    W,B,A= 0,0,0
    W,B = initialize(nodes, X.shape[0]) 
    return predict(W,B,A,sigmoid)

def cost(X, Y, Yp):
    m = X.shape[1]
    cost = -1/m*(np.sum(np.multiply(Y, np.log(Yp)) + np.multiply(1-Y, np.log(1-Yp)), axis=0))
    return cost

print(cost(X,Y,fp(X))) # using zeros for W the result should be 0.69

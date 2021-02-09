# multiclass
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

"""
To run we need:
    X: features
    Y: labels
"""
csv = pd.read_csv('winequality-white.csv', delimiter=';')

def datasets(csv):
    rows = csv.shape[0]
    sliceAt = int(np.sqrt(rows))
    Y = csv.iloc[sliceAt:,-1]
    X = csv.iloc[sliceAt:,:-1]
    X_test = csv.iloc[:sliceAt,:-1]
    Y_test = csv.iloc[:sliceAt,-1]
    return X,Y,X_test,Y_test

def zero_one(arr):
    """
    arr is an array of integers
    returns an array of 0s 1s for each integer
    with a one on the position. Example:
    [1,2] => [[1,0], [0,1]]
    """
    newArr=[]
    arrLen = max(arr)
    for el in arr:
        use = np.zeros(arrLen)
        for j in range(1,arrLen+1):
            if el == j:
                use[j-1] = 1
                newArr.append(use)
    return newArr

def to_numpy(X,Y,X_test,Y_test):
  X = X.to_numpy().T
  Y = Y.to_numpy().T
  X_test = X_test.to_numpy().T
  Y_test = Y_test.to_numpy().T
  return X,Y,X_test,Y_test

# forward propagation
def sigmoid(z):
    # returns predictions
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
    X features
    nodes nodes
    returns prediction
    """
    W,B,A= 0,0,0
    W,B = initialize(nodes, X.shape[0]) 
    return predict(W,B,A,sigmoid)

def cost(X, A, Ap):
    m = X.shape[1]
    cost = -1/m*(np.sum(np.multiply(A, np.log(Ap)) + np.multiply(1-A, np.log(1-Ap)), axis=0))
    print(cost.shape)
    return cost

X,Y,c,d = datasets(csv)
X,Y,c,d = to_numpy(X,Y,c,d)
Y = zero_one(Y)
print(Y)
#print(cost(X,Y,fp(X, 12))) # using zeros for W the result should be 0.69

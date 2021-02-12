# regression or classification with 1 out layer
import numpy as np

sigmoid = lambda z: 1/(1+np.exp(-z))
linear = lambda z: z

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

# forward propagation
def predict(W,B,X,m,activation):
    """
    W matrix/vector of weights
    B vector/number of biases
    X features
    fn function applied to the linear transform

    Returns the predictions (matrix or vector)
    """
    Z = np.dot(W,X) + B
    return activation(Z)

def cost(X, A, Ap, m, activation):
    if activation==sigmoid:
        return -1/m*(np.sum(A*np.log(Ap) + (1-A)*np.log(1-Ap), axis=1, keepdims=True))
    else:
        """there may be else if between, for other fns"""
        diff  = A-Ap
        return 1/m*np.dot(diff, diff.T)

# backpropagation
def update(W,B,X,A,Ap,m,lr):
    c = (1/m)*lr
    diff = c*(A-Ap)
    W = W + np.dot(diff,X.T)
    B = B + diff
    return W, B

def model(X,A,it=100,activation=sigmoid,lr=0.001):
    """ 
    X: matrix of features x samples
    A: matrix or vector of labels
    it: number of update iterations
    activation: for the out layer
    lr: learning rate
    """
    nodes, features, m = A.shape[0], X.shape[0], X.shape[1]
    # W,B for the 1st calculation
    W,B = initialize(nodes, features) 

    for i in range(it):
      Ap = predict(W,B,X,m,activation)
      W,B = update(W,B,X,A,Ap,m,lr)
      if i%(it/10)==0:
          print("Cost cycle %i: "%i,cost(X,A,Ap,m,activation))
    return W,B


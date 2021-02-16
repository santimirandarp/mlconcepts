# regression or classification with 1 out layer
import numpy as np

sigmoid = lambda z: 1/(1+np.exp(-z))
linear = lambda z: z

def initialize(nodes,features, itype="z"):
    """
    the rows of W are nodes
    the columns of W are features
    the rows of B are nodes

    returns W, B
    """
    print("number of nodes: ", nodes)
    print("number of features: ", features)
    if itype=="r":
      w = np.random.rand(nodes, features)
      b = np.random.rand(nodes, 1)
      return w, b
    elif itype == "z":    
      w = np.zeros((nodes, features))
      b = np.zeros((nodes, 1))
      return w, b

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

def cost(A, Ap, m, activation="sigmoid"):
    """
    A:labels, 
    Ap:predicted labels,
    m: N samples,
    activation: str sigmoid/linear, etc
    """
    if activation=="sigmoid":
        return -1/m*(np.sum(A*np.log(Ap) + (1-A)*np.log(1-Ap), axis=1, keepdims=True))
    elif activation=="linear":
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

def model(X,A,it=100,activation=sigmoid,lr=0.01):
    """ 
    X: matrix of features x samples
    A: matrix or vector of labels
    it: number of update iterations
    activation: for the out layer
    lr: learning rate
    """
    nodes, features, m = A.shape[0], X.shape[0], X.shape[1]
    print(nodes, features, m)
    # W,B for the 1st calculation
    W,B = initialize(nodes, features) 
    print(W.shape, B.shape)
    for i in range(it):
      Ap = predict(W,B,X,m,activation)
      W,B = update(W,B,X,A,Ap,m,lr)
      if i%(it/10)==0:
          print("Cost cycle %i:\n"%i,cost(X,A,Ap,m,activation))
    return W,B,Ap

def naive_estim(Y, method="regression", dtype='df'):
    """
    use this as a baseline
    if model has larger error => useless
    Y column vector
    method: regression or classification, string
    dtype: ndarray or df, string
    """
    l = Y.shape[0]
    if method == "classification" and dtype=='df':
        distribution = (Y.value_counts()/l)*100
        return distribution
    elif method == "regression":
        diff = Y-np.mean(Y)
        rmse = np.sqrt(np.dot(diff,diff.T)/l)
        return rmse


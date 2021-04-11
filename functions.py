# regression or classification with 1 out layer
import numpy as np

# FORWARD PROP
sigmoid = lambda z: 1/(1+np.exp(-z))
identity = lambda z: z


def normalize(X, axis=0):#axis=0 is like a line moving over cols
    """X could be np array or pandas"""
    return (X-X.mean(axis=axis))/(X.max(axis=axis)-X.min(axis=axis))


def initialize(nodes,features,fill="zeros"):
    """
    the rows of W are nodes
    the columns of W are features
    the rows of B are nodes
    returns W, B
    """

    print("number of nodes: ", nodes)
    print("number of features: ", features)

    if fill == "random":
      w = np.random.rand(nodes, features)
      b = np.random.rand(nodes, 1)
      return w, b

    elif fill == "zeros":    
      w = np.zeros((nodes, features))
      b = np.zeros((nodes, 1))
      return w, b

def predict(W,B,X,activation):

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
    activation: str sigmoid/identity, etc
    """
    if activation==sigmoid:
        return -1/m*(np.sum(A*np.log(Ap) + (1-A)*np.log(1-Ap), 
            axis=1, keepdims=True))

    elif activation==identity:
        """there may be else if between, for other fns"""
        diff  = A-Ap
        return 1/m*np.dot(diff, diff.T)


# BACKPROP

def update(w,b,X,diff,m,lr):
    c = (1/m)*lr
    diff = c*diff
    w += np.dot(diff,X.T)
    b += np.sum(diff,axis=1,keepdims=True)
    return w, b



# model
def fit_generic(X,A,it=100,activation=sigmoid,lr=0.1):
    """ 
      X: matrix of features x samples
      A: matrix or vector of labels
      it: number of update iterations
      activation: for the out layer, sigmoid or linear
      lr: learning rate
    """

    nodes, features, m = A.shape[0], X.shape[0], X.shape[1]

    W,B = initialize(nodes, features,"random") 

    for i in range(it):
      Ap = predict(W,B,X,activation)
      diff = A-Ap
      W,B = update(W,B,X,diff,m,lr)
      if i%(it/10)==0:
          print("Cost cycle %i:\n"%i,cost(A,Ap,m,activation))
    return W,B

def naive_estim(Y, method="regression"):
    """
      use this as a baseline
      if model has larger error => useless
      Y column vector
      method: regression or classification, string
      dtype: ndarray or df, string
    """

    l = Y.shape[1]
    if method == "classification":
        distribution = (Y.value_counts()/l)*100
        return distribution

    elif method == "regression":
        diff = Y-np.mean(Y)
        rmse = np.sqrt(np.dot(diff,diff.T)/l)
        return rmse


def metric(X,Y,w,b,method="regression"):
    """
      should be better than naive estimation
      if we have larger error it's useless

      method: regression or classification
    """
    m = X.shape[1]
    if method=="regression":
        Yp = predict(w,b,X,identity)
        diff = Y-Yp
        rmse = np.sqrt(np.dot(diff, diff.T)/m)
        return rmse

    elif method=="classification":
        Yp = predict(w,b,X,sigmoid)
        Yp = np.where(Yp > 0.5,1,0)
        Yp_inv = np.where(Yp==1,0,1)
        Y_inv = np.where(Y==1,0,1)
        accuracy = np.dot(Y,Yp.T)+np.dot(Y_inv,Yp_inv.T)
        print("correct/total", accuracy/m)
        return accuracy

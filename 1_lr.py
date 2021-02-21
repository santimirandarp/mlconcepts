# Linear Regression as described on the docs
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from csvfns import *
from functions import naive_model

sw = pd.read_csv('swedish_kronor.csv', delimiter='\t', decimal=',')

"""
    Next steps keep the columns,
    but slices sw (table) and normalizes
    features
"""
X,Y = toXY(sw, Ykey="Y")
X = normalize(X)
Y = normalize(Y) # don't need this for classifications (0 and 1).

" splits to create test and train dataframes"
X, Y, X_test, Y_test = trainAndTest(X,Y)
print(X,Y)
" convert dataframes to numpy (easier) and traspose"
X, Y, X_test, Y_test = to_numpy(X, Y, X_test, Y_test)

"traspose and reshape"
X = X.T # (features, samples)
X_test = X_test.T
Y = Y.reshape(1, Y.shape[0]) #this is a problem in to_numpy()
Y_test = Y_test.reshape(1, Y_test.shape[0]) #this is a problem in to_numpy()

" sanity check"
print("X train and test shape", X.shape, X_test.shape)
print("Y train and test shape", Y.shape, Y_test.shape) 
print("original shape was", sw.T.shape)

# FORWARD PROP (following a computation graph)
def initialize(dim):
    # dm is X.shape[0]
    w = np.zeros((1, dim))
    b = 0
    return w, b

def predict(X,w,b):
    #we need w,b, comes from initialize
    #returns Yp
    Yp = np.dot(w,X)+b 
    return Yp

def cost(X, Y, Yp):
    """
    X input data, (features x samples) array
    Y labels
    Yp predictions, same shape than Yp
    returns cost and difference Y-Yp
    """
    m = X.shape[1] 
    diff = Y - Yp 
    cost = 1/m*np.dot(A,A.T)
    return diff, cost

# BACKPROP
def update(X,diff,w,b,lr=0.1):
    """
      X input data, (features x samples) array
      diff: between labels - predictions, (1,samples) array
      w,b: weight and bias matrices
      lr: learning rate, a floating number
      returns new weight and bias
    """
   m = X.shape[1]
   grad = 2/m*np.sum(diff*X, axis=1, keepdims=True)
   dw = grad*lr
   w = w + dw 
   b = b + 2/m*np.sum(diff)*lr
   return w, b

def fit(X, Y, numIt):
    """
    X input data, (features x samples)
    Y labels
    numIt number of iterations for gradient descent
    """
    w,b = initialize(X.shape[0])
    Xflat, Yflat = X.flatten(), Y.flatten()
    plt.scatter(Xflat, Yflat, label="original")
    plt.title("Fit over cycles")
    diff,c = 0,0
    for i in range(numIt):
        Yp = predict(X,w,b)
        diff, c = cost(X,Y,Yp)
        w,b = update(X,diff,w,b)
        if i%(numIt/10)==0:
            plt.plot(Xflat,Yp.flatten(),label="it: "+str(i))
    plt.legend(loc="best")
    plt.show()
    print("cost: ", c)
    return w, b

def metric(X,Y,w,b):
    """
      calculate rmse
    """
    Yp = np.dot(w,X) + b
    diff = Y-Yp
    m = X.shape[1]
    rmse = np.sqrt(np.dot(diff, diff.T)/m)
    return rmse

print('naive estimation rmse: ', naive_model(Y))
w, b = fit(X, Y, 5000)
print("predictions rmse: ", metric(w,b,X,Y))

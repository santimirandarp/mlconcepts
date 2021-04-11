# Linear Regression as described on the docs
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from csvfns import *
from tabulate import tabulate
from functions import naive_estim, normalize

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

" convert dataframes to numpy (easier) and traspose"
X, Y, X_test, Y_test = to_numpy(X, Y, X_test, Y_test)

"traspose and reshape"
X = X.T # (features, samples)
X_test = X_test.T
Y = Y.reshape(1, Y.shape[0]) # (1 x samples)
Y_test = Y_test.reshape(1, Y_test.shape[0])

" shape check"
print("X train and test shape", X.shape, X_test.shape)
print("Y train and test shape", Y.shape, Y_test.shape) 
print("original shape was", sw.T.shape)


# FORWARD PROP (following a computation graph)

def initialize(dim):
    # dim is number of features
    w = np.zeros((1, dim))
    b = 0
    return w, b

def predict(X,w,b):
    # X inputs, (features x samples)
    # w,b from initialize
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
    cost = 1/m*np.dot(diff,diff.T)
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
    k = lr/m 
    dw = np.sum(diff*X, axis=1, keepdims=True)*k
    w += dw 
    b += np.sum(diff)*k
    return w, b

def fit(X, Y, numIt, plot=False):
    """
      X input data, (features x samples)
      Y labels
      numIt number of iterations for gradient descent
    """

    w,b = initialize(X.shape[0])
    Xflat, Yflat = X.flatten(), Y.flatten()
    plt.scatter(Xflat, Yflat, label="train", c="green")
    plt.title("Fit over cycles")
    diff,c = 0,0

    for i in range(numIt):
        Yp = predict(X,w,b)
        diff, c = cost(X,Y,Yp)
        w,b = update(X,diff,w,b)
        if i%(numIt/10)==0:
            plt.plot(Xflat,Yp.flatten(),label="it: "+str(i))
    plt.legend(loc="best")
    if plot: plt.show()
    print("cost: ", c)
    return w, b

def rmse(X,Y,w,b):
    """
      calculate rmse
    """
    Yp = predict(X,w,b)
    diff = Y-Yp
    m = X.shape[1]
    rmse = np.sqrt(np.dot(diff, diff.T)/m)
    return rmse


def run(X=X, Y=Y, X_test=X_test, Y_test=Y_test, numIt=5000, lr=1, metric=rmse, plot=True):

    w, b = fit(X, Y, numIt) # initializes, predicts and updates (iter)

    naive_rmse_train = naive_estim(Y)
    rmse_train = metric(X,Y,w,b)

    naive_rmse_test = naive_estim(Y_test)
    rmse_test = metric(X_test,Y_test,w,b)

    plt.scatter(X_test, Y_test, label="test", c="red") 
    plt.legend(loc="best")
    if plot: plt.show()

    table = [
            ["naive_rmse_train: ", naive_rmse_train],
            ["rmse_train: ", rmse_train],
            ["naive_rmse_test: ", naive_rmse_test],
            ["rmse_test: ", rmse_test]
            ]
    print(tabulate(table))

run()

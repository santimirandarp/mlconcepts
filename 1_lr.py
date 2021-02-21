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

print(X)

" splits to create test and train dataframes"
X,Y,X_test,Y_test = trainAndTest(X,Y)

" convert dataframes to numpy (easier) and traspose"
X,Y,X_test,Y_test = to_numpy(X,Y,X_test,Y_test, traspose=True)

print("X shape is", X.shape, "Y shape is", Y.shape)


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
    # returns cost
    m = X.shape[1] 
    A = Y - Yp 
    cost = 1/m*np.dot(A,A.T)
    return A, cost


# backward propagation
def update(X,A,w,b,lr=0.1):
   m = X.shape[1]
   grad = 2/m*np.sum(A*X, axis=1, keepdims=True)
   dw = grad*lr
   w = w + dw 
   b = b + 2/m*np.sum(A)*lr
   return w, b

def model(X, Y, numIt):
    w,b = initialize(X.shape[0])
    Xflat=X.flatten()
    Yflat=Y.flatten()
    plt.scatter(Xflat, Yflat, label="original")
    plt.title("Fit over cycles")
    A,c = 0,0
    for i in range(numIt):
        Yp = predict(X,w,b)
        A, c = cost(X,Y,Yp)
        w,b = update(X,A,w,b)
        if i%(numIt/10)==0:
            plt.plot(Xflat,Yp.flatten(),label="it: "+str(i))
    plt.legend(loc="best")
    plt.show()
    print("cost: ", c)
    return w, b

def metric(w,b,X,Y):
    Yp = np.dot(w,X) + b
    diff = Y-Yp
    l = Y.shape[1]
    rmse = np.sqrt(np.dot(diff, diff.T)/l)
    return rmse

print('naive estimation rmse: ', naive_model(Y))
w, b = model(X, Y, 500)
print("model rmse: ", metric(w,b,X,Y))

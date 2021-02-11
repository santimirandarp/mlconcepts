# Linear Regression as described on the docs
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# import swedish kronor dataset
sw = pd.read_csv('swedish_kronor.csv', delimiter='\t', decimal=',')

# convert dataframe to numpy.arr to get +functionality
def make_datasets(dataset):
    X, Y = sw['X'].to_numpy(), sw['Y'].to_numpy()
    X, Y = X.reshape(1,X.shape[0]), Y.reshape(1,Y.shape[0]) 
    # from column to row
    return X, Y

def naive_model(Y):
    """
    use this as a baseline
    if we have larger error it's useless
    """
    l = Y.shape[1]
    diff = Y-np.mean(Y)
    rmse = np.sqrt(np.dot(diff,diff.T)/l)
    return rmse

def normalize(X):
    """
    X features, returns normalized features
    """
    return (X - np.mean(X))/(np.max(X)-np.min(X))


###### forward propagation

def initialize(dim):
    # dim is X.shape[0]
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

X,Y = make_datasets(sw)
print('naive estimation rmse: ', naive_model(Y))
X = normalize(X)
w, b = model(X, Y, 500)
print("model rmse: ", metric(w,b,X,Y))

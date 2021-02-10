# Linear Regression as described on the docs
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# import swedish kronor dataset
sw = pd.read_csv('swedish_kronor.csv', delimiter='\t', decimal=',')

# convert dataframe to numpy.arr to get +functionality
def make_datasets(X,Y):
    X, Y = sw['X'].to_numpy(), sw['Y'].to_numpy()
    X, Y = X.T, Y.T # from colum to row
    return X, Y

def naive_model(Y):
    """
    use this as a baseline
    if we have larger error it's useless
    """
    samples = Y.shape[1]
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

print('naive estimation', naive_model(Y))
X = normalize(X)
w, b = model(X, Y, 500)

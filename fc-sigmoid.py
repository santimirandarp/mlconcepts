# Logistic Regression as described on the docs
import numpy as np 

X = np.array([[10],[20],[30],[40]]) # 4 examples, 1 feature
X = (X - np.mean(X))/(max(X)-min(X))
Y = np.array([[1.01,2.02,4,6]])

# forward propagation

def sigmoid(z):
    # vector of predictions
    return 1/(1+np.exp(z))

def predict(w, b, X):
    # returns Yp
    z = np.dot(w,X.T) + b
    return sigmoid(z)

def cost(X, Y, Yp):
    m = X.shape[1] 
#    cost = 1/m*(np.dot(Y, np.log(Yp).T) + np.dot(1-Y, np.log(1-Yp).T))
    cost = 1/m*(np.sum(np.multiply(Y, np.log(Yp)) + np.multiply(1-Y, np.log(1-Yp))))
    return cost

def initialize(dim):
    # dim is X.shape[0]
    w = np.zeros((1, dim))
    b = 0
    return w, b

def fp(X):
    w, b = initialize(X.shape[1])
    Yp = predict(w, b, X)
    return cost(X,Y,Yp)

print("cost is", fp(X))

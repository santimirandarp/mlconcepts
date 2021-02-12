# shallow network
import numpy as np 
import matplotlib.pyplot as plt

X = np.array([[11,20,30,40,50,80,90,100]]) 
X = (X - np.mean(X))/(np.max(X)-np.min(X))
Y = np.array([[1,1,1,1,1,0,0,0]]) 

# forward propagation

def sigmoid(z):
    # vector of predictions
    return 1/(1+np.exp(-z))

def initialize(nodes,features):
    w = np.zeros((nodes, features))
    #each node computes for a set of features
    b = np.zeros((nodes, 1))
    return w, b

def predict(W,B,X,fn):
    A = np.dot(W,X) + B
    return fn(A)

def fp(X):
    arch= [X.shape[0],4,3,2,1] #number of nodes for each layer
    W,B,A= 0,0,0
    for layer in arch[0:-1]:
        W,B = initialize(arch[layer+1], arch[layer]) 
        A = predict(W,B,A,np.tanh) # 2, m
    W,B = initialize(arch[-1], A.shape[0]) #nodes x features
    return predict(W,B,A,sigmoid)

def cost(X, Y, Yp):
    m = X.shape[1]
    cost = -1/m*(np.sum(np.multiply(Y, np.log(Yp)) + np.multiply(1-Y, np.log(1-Yp))))
    return cost

print(cost(X,Y,fp(X))) # using zeros for W the result should be 0.69

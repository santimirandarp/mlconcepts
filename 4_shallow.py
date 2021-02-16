# shallow network
import numpy as np 
import matplotlib.pyplot as plt
from functions import *

X = np.array([[11,20,30,40,50,80,90,100]]) 
A0 = (X - np.mean(X, axis=1, keepdims=True))/(np.max(X)-np.min(X))
Y = np.array([[1,1,1,1,1,0,0,0]]) 

# check initialize is random

def fp(A0,m,nodes=[4,3,2,1],hfn=np.tanh,ofn=sigmoid):
    """
    A0, matrix of samples
    nodes: architecture, list of nodes per layer
    hfn: hidden layer function
    ofn: output layer function
    returns row vector/matrix of predictions
    """
    W,B = initialize(nodes[0], A0.shape[0]) 
    Ap = predict(W,B,A0,m,hfn) # nodesL x samples
    for l in range(len(nodes[1:-1])):
        W,B = initialize(nodes[l+1], nodes[l]) 
        Ap = predict(W,B,Ap,m,hfn) # nodes x samples 
    W,B = initialize(nodes[-1], Ap.shape[0]) #nodes x nodes_prev
    return predict(W,B,Ap,m,ofn)

m = A0.shape[1]
print(cost(Y,fp(A0=X, m=m),m=m,activation="sigmoid")) 

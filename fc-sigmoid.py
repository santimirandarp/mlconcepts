# Logistic Regression as described on the docs
import numpy as np 
import matplotlib.pyplot as plt

X = np.array([[11,20,30,40,50,80,90,100]]) # 4,1
#X = (X - np.mean(X))/(np.max(X)-np.min(X))
Y = np.array([[1,1,1,1,1,0,0,0]]) #1,4

# forward propagation

def sigmoid(z):
    # vector of predictions
    return 1/(1+np.exp(-z))

def initialize(dim):
    # dim: n of features
    w = np.zeros((1, dim))
    b = 0
    return w, b

def predict(w, b, X):
    # returns Yp
    z = np.dot(w,X) + b
    return sigmoid(z)

def cost(X, Y, Yp):
    m = X.shape[1]
    cost = -1/m*(np.sum(np.multiply(Y, np.log(Yp)) + np.multiply(1-Y, np.log(1-Yp))))
    return cost


# backwards propagation
def update(w,b,X,Y,Yp,step=10):
    m = X.shape[1]
    diff = Y-Yp
    grad = 1/m*np.sum(diff*X, axis=1, keepdims=True)
    w += grad*step
    b += 1/m*np.sum(diff)*step
    return w,b

def model(X,Y,cycles=100):
   w,b = initialize(X.shape[0])
   m = X.shape[1]
   plt.scatter(X,Y)
   for i in range(cycles):
       Yp = predict(w,b,X)
       w,b = update(w,b,X,Y,Yp)
       if i%(cycles/10)==0:
           print(cost(X,Y,Yp))
           plt.plot(X.flatten(),Yp.flatten(), label="it: "+str(i))
           plt.legend(loc="upper right")
   plt.show()
   return w,b

model(X,Y)


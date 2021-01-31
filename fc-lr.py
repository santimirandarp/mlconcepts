# Linear Regression as described on the docs
import numpy as np 
import matplotlib.pyplot as plt

#X = np.array([[1.01,2.01,3.05,4], [0.98,2.05,3.1,5]])
X = np.array([[10],[20],[30],[40]])
X = (X - np.mean(X))/(max(X)-min(X))
Y = np.array([[1.01,2.02,2.88,4.1]])
### (4,1) (1, 4)

# forward propagation
def cost(X, Y, Yp):
    # we need Yp, comes from predict
    # returns cost
    m = X.shape[1] #4
    A = Y - Yp # (1, m)
    cost = 1/m*np.dot(A,A.T).flatten()
    return A, cost

def predict(X,w,b):
    #we need w,b, comes from initialize
    #returns Yp
    Yp = np.dot(w,X.T)+b # 1x4
    return Yp

def initialize(dim):
    # dim is X.shape[0]
    w = np.zeros((1, dim))
    b = 0
    return w, b

# backward propagation
def update(X,A,w,b,lr=0.01):
   m = X.shape[1]
   dw = 1/m*np.dot(2*A, X)*lr
   w = w + dw 
   b = b + np.sum(1/m*2*A*lr)
   return w, b

def model(X, Y, numIt):
    w,b = initialize(X.shape[1])
    plt.scatter(X.flatten(), Y.flatten(), label="original")
    plt.title("Fit line over cycles, features norm")
    for i in range(numIt):
        Yp = predict(X,w,b)
        A, c = cost(X,Y,Yp)
        w,b = update(X,A,w,b,lr=0.01)
        if i%10==0:
            plt.plot(X.flatten(),Yp.flatten(), label="it: "+str(i))
    plt.legend(loc="best")
    plt.show()
    print("cost: ", c)
    return w, b

w, b = model(X,Y,50)
#print("printing w,b", w,b)
#X = [[1.9],[2.11]]
#Yp = np.dot(w,X)+b # 1x4
#print(Yp)

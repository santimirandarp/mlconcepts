# Logistic Regression as described on the docs
import numpy as np 
import matplotlib.pyplot as plt
from functions import *

X = np.array([[11,20,30,40,50,80,90,100]]) 
X = normalize(X,axis=1)
Y = np.array([[1,1,1,1,1,0,0,0]]) 
#all are 1x8

def model(X,Y,numIt=500,lr=1):

   ft, m = X.shape[0], X.shape[1]
   outnode = 1
   w,b = initialize(outnode,ft) # 1x1 both

   for i in range(numIt):
       Yp = predict(w,b,X,sigmoid)
       diff = Y-Yp
       w,b = update(w,b,X,diff,m,lr)
       print(w,b)
       if i%(numIt/10)==0:
           print(cost(Y,Yp,m,"sigmoid"))
           plt.plot(X.flatten(),Yp.flatten(), label="it: "+str(i))
           plt.legend(loc="upper right")

   plt.scatter(X,Y)
   plt.show()
   return w,b

model(X,Y)

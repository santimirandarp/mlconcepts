# Logistic Regression as described on the docs
import numpy as np 
import matplotlib.pyplot as plt
from functions import *

"""data to set up the model"""

X = np.array([[11,20,30,40,50,80,90,100]]) 
X = normalize(X,axis=1)
Y = np.array([[1,1,1,1,1,0,0,0]]) 
# all are 1x8

def fit(X,Y,numIt=500,lr=1):

   ft, m = X.shape[0], X.shape[1]
   outnode = 1
   w,b = initialize(outnode,ft) # 1x1 both

   for i in range(numIt):
       Yp = predict(w,b,X,sigmoid)
       diff = Y-Yp
       w,b = update(w,b,X,diff,m,lr)

       if i%(numIt/10)==0:
           print(cost(Y,Yp,m,"sigmoid"))
           plt.plot(X.flatten(),Yp.flatten(), label="it: "+str(i))
           plt.legend(loc="upper right")

   plt.scatter(X,Y,c="green")
   plt.show()
   return w,b

w,b = fit(X,Y,numIt=1000)

"""we need to evaluate performance of the model"""
acc_train=metric(X,Y,w,b,"classification")

#acc_test=metric(X_test,Y_test,w,b,"classification")
"""don't have test set in this case"""

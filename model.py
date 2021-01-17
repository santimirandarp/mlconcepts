import ds
import numpy as np 

"""
Binary Regression
  Setup
"""

# Import Data
datasets="images/clean/datasets.hdf5"
train_X, train_Y, test_X, test_Y, classes = ds.load(datasets)


# Make each image a column
train_X = train_X.reshape(train_X.shape[0], -1).T 
test_X = test_X.reshape(test_X.shape[0], -1).T


# Normalize pixels
train_X = train_X/255
test_X = test_X/255
print(test_X)

"""
Forward Propagation
  Cost = -1/m*sum(log-loss{sigmoid...})
  (measure error)
"""

def sigmoid(z):
    """
    z a vector or number
    """
    return 1/(1 + np.exp(-z))


def cost(m, Yp, Y):
    """
    samples, Y predicted, Y real
    returns number
    """
    return -1/m*np.sum(Y*np.log(Yp) + (1-Y)*np.log(1-Yp)) 

"""
Backward propagation
  Compute the gradients
  Update w and b
"""

def gradients(m, X, Yp, Y):
    """
    Features, Y predicted, Y real
    """
    m = X.shape[0]
    err = Yp-Y # 1, 276
    dw = 1/m*np.dot(X, err.T)
    db = 1/m*np.sum(err)
    return dw, db

def update(m, w, b, X, Yp, Y, rate):
    "gradient descent"
    dw, db = gradients(m, X, Yp, Y)
    w -= dw*rate
    b -= db*rate 
    return w, b

def init_wb(dim):
    """
    input: samples' size
    """
    w = np.zeros((dim, 1))
    b = 0
    return w, b

def model(X, Y, cycles=2000, rate=0.5):
   """
   Features, Y real
   """
   m, pixels = X.shape[1], X.shape[0]
   w, b = init_wb(pixels) 
   for i in range(cycles):
       Yp = sigmoid(np.dot(w.T, X)+ b) 
       err = cost(m,Yp,Y)
       print(err)
       w, b = update(m,w,b,X,Yp,Y,rate)
   return w, b 

"""
Predict
    The model yields the best W and B
    We use them to predict
"""
w, b = model(train_X, train_Y)


def predict(w, b, test_X, test_Y, classes):
   pred = sigmoid(np.dot(w.T, test_X)+b)
   correct = 0
   for i in range(len(pred[0])):
       thisPred = pred[0,i]
       if(thisPred > 0.5):
           thisPred = 1
       else:
           thisPred= 0
       prediction = classes[thisPred]
       real = classes[test_Y[0,i]]
       print("you predicted", prediction)
       print("it is a", real)
       if(prediction==real):
          correct+=1
   print("correct/total", (correct/24)*100) 
predict(w,b,test_X,test_Y,classes)

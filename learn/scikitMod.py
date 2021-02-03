import sklearn
import sklearn.datasets # datasets for free!
import sklearn.linear_model
import ds
import numpy as np 
import matplotlib.pyplot as plt

# Import Data
datasets="images/clean/datasets.hdf5"
train_X, Y, a, b, c= ds.load(datasets)

# Make each image a column
train_X = train_X.reshape(train_X.shape[0], -1) 

# Normalize pixels
X = train_X/255

clf = sklearn.linear_model.LogisticRegressionCV(max_iter=2000);
print(X.shape, Y.T.shape)
print(clf.fit(X, np.ravel(Y.T)))

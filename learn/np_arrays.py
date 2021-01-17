import numpy as np

# 10 pictures; 4x4 pixels, 3 (r,g,b) values each
narray = np.arange(10*4*4*3) 
print("prediction 480")
print("narray is", ndarray.shape) 

matrix= narray.reshape(10, 4, 4, 3) # matrix
print("prediction is 10,4,4,3")
print("matrix shape is", matrix.shape) 

# if a specific axis is specified to .reshape, that axis is preserved and the rest of the axis' are reshaped/flattened
# a -1 argument tells numpy to figure out the dimensions of reshaped axis 

# flatten the innermost axis (the r,g,b values), which are already flat, so this operation does nothing
a = matrix

aflat=a.reshape(a.shape[0],a.shape[1],a.shape[2],-1)
print(aflat.shape)
print(aflat)

# flatten the innermost two axis (r,g,b values in each pixel row). 4x3 gets flattened to 12 color values
aflat=a.reshape(a.shape[0],a.shape[1],-1)
print("prediction: 10, 4, 12") 
print(aflat.shape) 
print(aflat)

# this operation flattens each individual image
aflat=a.reshape(a.shape[0],-1) 
print("prediction: 10, 48") 
print(aflat.shape)
print(aflat)

# the rows have 'examples'  and columns have the 'features', we transpose the matrix using the .T method
aflatt=aflat.T
print("prediction: 48, 10") 
print(aflatt.shape)
print(aflatt)

# fun exercise
# to create random pixel noise to test your trained network, try the following
# x_test=np.random.randint(255,size=(64*64*3,209))
# print(x_test.shape)
# print(x_test)

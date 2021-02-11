import numpy as np
# generates datasets
def datasets(df_csv):
    # get X and normalize
    ft = df_csv.iloc[:,:-1]
    ft = (ft-ft.mean())/(ft.max()-ft.mean())
    print(ft.head()) #just to see the ft normalized
    rows = df_csv.shape[0]
    sliceAt = int(np.sqrt(rows))
    Y = df_csv.iloc[sliceAt:,-1]
    Y_test = df_csv.iloc[:sliceAt,-1]
    X = ft.iloc[sliceAt:,:]
    X_test = ft.iloc[:sliceAt,:]
    return X,Y,X_test,Y_test

def zero_one(labels):
    """
    labels is an array of integers
    returns an array of 0s 1s for each integer
    with a one on the position. Example:
    [1,2] => [[1,0], [0,1]]
    """
    newArr=[]
    arrLen = max(labels) # 9
    for el in labels:
        use = np.zeros(arrLen+1) # 9, from 0 to 8
        for j in range(0,arrLen+1):
            if el == j:
                use[j] = 1
                newArr.append(use)
    newArr = np.array(newArr).T
    return newArr

def to_numpy(X,Y,X_test,Y_test):
    """takes the pandas datasets
    retrieves numpy arrays"""
    X = X.to_numpy().T
    Y = Y.to_numpy().T
    X_test = X_test.to_numpy().T
    Y_test = Y_test.to_numpy().T
    return X,Y,X_test,Y_test

def sigmoid(z):
    """ takes linear piece
    returns the prediction"""
    return 1/(1+np.exp(-z))

def initialize(nodes,features):
    """
    the rows of W are nodes
    the columns of W are features
    the rows of B are nodes

    returns W, B
    """
    w = np.random.rand(nodes, features)
    #each node computes for a set of features
    b = np.random.rand(nodes, 1)
    return w, b


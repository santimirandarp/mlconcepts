import numpy as np

"""Set of utilities to prepare the data"""

def toXY(table, Ykey=False, Yindex=False):
    """
    returns X,Y as dataframes
    table: from csv_read, dataframe
    Ykey: string to access df's Y key
    Yindex: alternative to Ykey. Can be + or -
    """
    Y,X = None, None
    if isinstance(Ykey, str):
        Y = table[Ykey]
        del table[Ykey]
    elif isinstance(Yindex, int):
        if Yindex != -1:
            Y = table.iloc[:, Yindex:Yindex+1]
            del table.iloc[:, Yindex:Yindex+1]
        elif Yindex==-1:
            Y = table.iloc[:, Yindex:]
            del table.iloc[:, Yindex:]
    else:
        print('Execution Error. Did you pass Ykey or Yindex?')
        return None
    X = table #where Y has been removed
    return X, Y

def to_numpy(X,Y,X_test,Y_test):
    """takes the pandas datasets
    retrieves numpy arrays"""
    X = X.to_numpy()
    Y = Y.to_numpy()
    X_test = X_test.to_numpy()
    Y_test = Y_test.to_numpy()
    return X,Y,X_test,Y_test 


def trainAndTest(X,Y, sliceAt = np.sqrt):
    """
    Assumes X shape (samples, features) as in a table
    Y (samples,1)
    sliceAt is the operation done over m
    lambda functions can be defined on the fly
    """
    m = X.shape[0] 
    sliceAt = int(sliceAt(m))
    Y = Y.iloc[sliceAt:]
    Y_test = Y.iloc[:sliceAt]
    X = X.iloc[sliceAt:,:]
    X_test = X.iloc[:sliceAt,:]
    return X,Y,X_test,Y_test


# utility for multiclass problems
def zero_one(labels):
    """
    labels is an array of integers
    returns an array of 0s 1s for each integer
    with a one on the position. Example:
    [1,2] => [[1,0], [0,1]]
    """
    newArr=[]
    labels=labels.flatten()
    maxV = int(max(labels))
    minV = int(min(labels))
    for el in labels:
        use = np.zeros(maxV-minV+1) 
        for j in range(minV,maxV+1):
            if int(el) == j:
                use[j-3] = 1
                newArr.append(use)
    newArr = np.array(newArr).T
    return newArr

import numpy as np
# generates datasets

# also built into datasets
def to_numpy(X,Y,X_test,Y_test):
    """takes the pandas datasets
    retrieves numpy arrays"""
    X = X.to_numpy()
    Y = Y.to_numpy()
    X_test = X_test.to_numpy()
    Y_test = Y_test.to_numpy()
    return X,Y,X_test,Y_test

def datasets(df_csv,normalize=True, sliceAt = np.sqrt, to_numpy=to_numpy):
    """from 2 columns (features and labels), 
    return test and train datasets.
    The last column has to be Y (labels)"""
    df_csv = df_csv.T # transpose
    ft = df_csv.iloc[:-1,:] # separate last row Y
    if normalize==True:
        #ft = (ft-ft.mean())/(ft.max()-ft.min())
        ft = (ft-ft.mean(axis=0))/(ft.max(axis=0)-ft.min(axis=0))
    print("Head: ",ft.head()) #just to see the ft normalized
    cols = df_csv.shape[1] 
    sliceAt = int(sliceAt(cols))
    Y = df_csv.iloc[-1,sliceAt:]
    Y_test = df_csv.iloc[-1,:sliceAt]
    X = ft.iloc[:,sliceAt:]
    X_test = ft.iloc[:,:sliceAt]
    if to_numpy==to_numpy:
        return to_numpy(X,Y,X_test,Y_test)
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
    arrLen = int(max(labels)) # 9
    for el in labels:
        use = np.zeros(arrLen+1) # 9, from 0 to 8
        for j in range(0,arrLen+1):
            if el == j:
                use[j] = 1
                newArr.append(use)
    newArr = np.array(newArr).T
    return newArr

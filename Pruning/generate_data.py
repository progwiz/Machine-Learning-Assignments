import numpy as np
import scipy.stats
import math

def generate_data(m):
    data=[]   
    for i in range(m):
        X=[]
        X1=np.random.uniform()
        if X1<0.5:
            X1=0
        else:
            X1=1
        X.append(X1)
        for j in range(14):
            x=np.random.uniform()
            if x<0.75:
                x=X[j]
            else:
                x=1-X[j]
            X.append(x)
        for j in range(6):
            x=np.random.uniform()
            if x<0.5:
                x=1
            else:
                x=0
            X.append(x)
        if X1==0:
            Y=scipy.stats.mode(X[1:8])[0][0]
        else:
            Y=scipy.stats.mode(X[8:15])[0][0]
        data.append([X,Y])
    return data
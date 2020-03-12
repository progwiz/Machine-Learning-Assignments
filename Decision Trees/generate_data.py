import numpy as np
import math

def generate_data(k,m):
    data=[]
    w_denominator=0
    for i in range(k-1):
        w_denominator+=(0.9**(i+2))        
    for i in range(m):
        X=[]
        X1=np.random.uniform()
        if X1<0.5:
            X1=0
        else:
            X1=1
        X.append(X1)
        y_sum=0
        for j in range(k-1):
            x=np.random.uniform()
            if x<0.75:
                x=X[j]
            else:
                x=1-X[j]
            X.append(x)
            w=(0.9**(j+2))/w_denominator
            y_sum+=w*x
        if y_sum>=0.5:
            Y=X1
        else:
            Y=1-X1
        data.append([X,Y])
    return data
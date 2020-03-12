import numpy as np
from sklearn import datasets, linear_model


if __name__ == "__main__":
    m=200
    w=1
    B=5
    sigma_square = 0.1
    W=[]
    W_dash=[]
    b=[]
    b_dash=[]
    for rep in range(1000):
        X=[]
        X_centered=[]
        Y=[]
        for _ in range(m):
            x=np.random.uniform(100,102)
            X.append(x)
            Y.append(x*w+B+np.random.normal(scale=sigma_square))
            X_centered.append(x-101)
        X=np.reshape(X,(-1,1))
        X_centered=np.reshape(X_centered,(-1,1))
        regr = linear_model.LinearRegression(fit_intercept=True)
        regr.fit(X,Y)
        W.append(regr.coef_[0])
        b.append(regr.intercept_)
        regr = linear_model.LinearRegression(fit_intercept=True)
        regr.fit(X_centered,Y)
        W_dash.append(regr.coef_[0])
        b_dash.append(regr.intercept_)

        
    
    exp_w=np.mean(W)
    exp_w_dash=np.mean(W_dash)
    exp_b=np.mean(b)
    exp_b_dash=np.mean(b_dash)
    
    print(exp_w,exp_b)
    print(exp_w_dash,exp_b_dash)
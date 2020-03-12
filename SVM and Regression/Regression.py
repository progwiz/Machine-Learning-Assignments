import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import warnings
import math
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")


def generate_data(m):
    Xs=[]
    Ys=[]
    for _ in range(m):
        X=[1]
        X+=list(np.random.normal(size=10))
        X.append(X[1]+X[2]+np.random.normal(scale=0.1**0.5))
        X.append(X[3]+X[4]+np.random.normal(scale=0.1**0.5))
        X.append(X[4]+X[5]+np.random.normal(scale=0.1**0.5))
        X.append(0.1*X[7]+np.random.normal(scale=0.1**0.5))
        X.append(2*X[2]-10+np.random.normal(scale=0.1**0.5))
        X+=list(np.random.normal(size=5))
        multipliers=[0.6**(i) for i in range(1,11)]
        Y=10+np.dot(multipliers, X[1:11])+np.random.normal(scale=0.1**0.5)
        Xs.append(X)
        Ys.append(Y)
    Xs=np.array(Xs)
    Xs=np.reshape(Xs,(m,21))
    Ys=np.array(Ys)
    Ys=np.reshape(Ys,(m,1))
    return Xs,Ys

def test_model(test_X,test_Y,Ws):
    predict_Y=np.matmul(test_X,Ws)
    return mean_squared_error(predict_Y,test_Y)

def simple_regression(X,Y):
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.transpose(X)),Y)

def ridge_regression(X,Y,l):
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)+l*np.identity(X.shape[1])),np.transpose(X)),Y)

def lasso_regression(X,Y,l, i=1000):
    
    w=np.full((X.shape[1],1),0,dtype=np.float64)
    for _ in range(i):    
        feature_no=np.random.randint(0,X.shape[1])
        
        if feature_no==0:
            #bias term update:
            w[0]=w[0]+np.mean(Y-np.matmul(X,w))
    
        else:
            #non-bias term update
            w_old=w[feature_no]
            feature_vector=np.array(X[:,feature_no])
            feature_vector=np.reshape(feature_vector,(X.shape[0],1))
            feature_vector_transpose=np.reshape(feature_vector,(1,X.shape[0]))
            term1=float(np.matmul(feature_vector_transpose,Y-np.matmul(X,w)))
            term2=float(l/2)
            term3=float(np.matmul(feature_vector_transpose,feature_vector))
            decider1=(-term1+term2)/term3
            decider2=(-term1-term2)/term3
            if decider1<w_old:
                w[feature_no]=w_old-decider1
            elif w_old<decider2:
                w[feature_no]=w_old-decider2
            else:
                w[feature_no]=0
        
    return w
    

if __name__ =="__main__":
    X,Y=generate_data(1000)
    test_X,test_Y=generate_data(1000000)
    
    Xmean=X[:,1:].mean(axis=0)
    Xstd=X[:,1:].std(axis=0)
    Ymean=Y.mean(axis=0)
    X[:,1:]=(X[:,1:]-Xmean)/Xstd
    Y=Y-Ymean
    
    test_X[:,1:]=(test_X[:,1:]-Xmean)/Xstd
    test_Y=test_Y-Ymean
    
    actual_w=[10-Ymean]
    actual_w+=[0.6**(i) for i in range(1,11)]
    actual_w+=[0 for i in range(11,21)]
    actual_w=np.reshape(actual_w,(21,1))
    
    print("Ymean", Ymean)
    print("Simple Regression------------------------")
    #Simple least square regression
    w_simple=simple_regression(X,Y)
    print("Weights using simple regression:",w_simple)
    error_simple=test_model(test_X,test_Y,w_simple)
    var_simple=np.var(abs(w_simple[1:]-actual_w[1:]))
    weight_simple=np.mean(abs(w_simple[1:]-actual_w[1:]))
    bias_simple=abs(w_simple[0]-actual_w[0])
    print("Simple regression error:", error_simple)
    print("Error in weights:", weight_simple)
    print("Variance in weights:", var_simple)
    print("Error in bias:", bias_simple)
    
    print("Ridge regression ------------------------------")
    
    ridge_errors=[]
    Ls=np.arange(0,10,0.05)
    minimum=9999
    for l in Ls:
        w_ridge=ridge_regression(X,Y,l)
        error=test_model(test_X,test_Y,w_ridge)
        ridge_errors.append(error)
        if error<minimum:
            W_ridge=w_ridge
            minimum=error
            optimal_L=l
    plt.figure(0)
    plt.title("Ridge regression Lambda v/s error")
    plt.xlabel("Lambda")
    plt.ylabel("Mean square error")
    plt.plot(Ls,ridge_errors)
    print("optimal Lambda:",optimal_L)
    print("Weights at optimal Lambda:",W_ridge)
    print("Ridge error:", minimum)
    print("Error in weights:", np.mean(abs(W_ridge[1:]-actual_w[1:])))
    print("Variance in weights:", np.var(abs(W_ridge[1:]-actual_w[1:])))
    print("Error in bias:", abs(W_ridge[0]-actual_w[0]))
    

    print("Lasso Regression-----------------------------")
    #Lasso regression
    lasso_errors=[]
    Ls=np.arange(0,1000,1)
    Zeros=[]
    minimum=9999
    for l in Ls:
        w_lasso=lasso_regression(X,Y,l)
        #print(w_lasso)
        error=test_model(test_X,test_Y,w_lasso)        
        lasso_errors.append(error)
        Zeros.append(20-len(list(np.transpose(np.nonzero(w_lasso[1:])))))
        if error<minimum:
            W_lasso=w_lasso
            minimum=error
            optimal_L=l

    plt.figure(11)
    plt.title("Lasso regression Lambda v/s error")
    plt.xlabel("Lambda")
    plt.ylabel("Mean square error")
    plt.plot(Ls,lasso_errors)
    
    print("optimal Lambda:",optimal_L)
    print("Weights at optimal Lambda:",W_lasso)
    print("Lasso error:", minimum)
    print("Error in weights:", np.mean(abs(W_lasso[1:]-actual_w[1:])))
    print("Variance in weights:", np.var(abs(W_lasso[1:]-actual_w[1:])))
    print("Error in bias:", abs(W_lasso[0]-actual_w[0]))
    
    plt.figure(12)
    plt.title("Lasso regression No. of pruned features")
    plt.xlabel("Lambda")
    plt.ylabel("No. of pruned features")
    plt.plot(Ls,Zeros)
    
    # COMPARISON -------------------------------
    print("Comparison---------------------------------")
    print("------------------------------------------")
    print("naive least squares-----------------------")
    test_X,test_Y=generate_data(1000000)
    test_X[:,1:]=test_X[:,1:]-Xmean/Xstd
    test_Y=test_Y-Ymean
    error_simple=test_model(test_X,test_Y,w_simple)
    var_simple=np.var(abs(w_simple[1:]-actual_w[1:]))
    weight_simple=np.mean(abs(w_simple[1:]-actual_w[1:]))
    bias_simple=abs(w_simple[0]-actual_w[0])
    print("Simple regression error:", error_simple)
    print("Error in weights:", weight_simple)
    print("Variance in weights:", var_simple)
    print("Error in bias:", bias_simple)
    
    print("combined -----------------------------------")
    relevant_features=np.nonzero(W_lasso)[0]
    relevant_W=[w for w in list(W_lasso[:,0]) if w!=0]
    X=X[:,relevant_features]
    test_X=test_X[:,relevant_features]
    actual_w=actual_w[relevant_features]
    w_ridge=ridge_regression(X,Y,optimal_L)
    error_ridge=test_model(test_X,test_Y,w_ridge)
    weight_ridge=np.sum(abs(w_ridge[1:]-actual_w[1:]))/20
    #var_ridge=np.var(abs(w_ridge[1:]-actual_w[1:]))
    bias_ridge=abs(W_ridge[0]-actual_w[0])
    print("Mean Square error: ", error_ridge)
    print("Error in weights:", weight_ridge)
    #print("Variance in weights:", var_ridge)
    print("Error in bias:", bias_ridge)
    print("Differences---------------------------------")
    print("Error difference", error_simple-error_ridge)
    print("Weight error difference", weight_simple-weight_ridge)
    #print("Weight var difference", var_simple-var_ridge)
    print("Bias difference", bias_simple-bias_ridge)

    
    
    
    
    
    
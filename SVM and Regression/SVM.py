import numpy as np

def inv(m):
    a, b = m.shape
    if a != b:
        raise ValueError("Only square matrices are invertible.")

    i = np.eye(a, a)
    return np.linalg.lstsq(m, i)[0]

def kernel(X1,X2):
    return (1+np.dot(X1,X2))**2

def newton_method(alphas,X,Y,eta):
    gradients=np.zeros((len(alphas)-1,1))
    alphay_sum=0
    for i in range(1,len(alphas)):
        alphay_sum+=alphas[i]*Y[i]
    for i in range(1,len(alphas)):
        teta=-eta*((1/alphas[i])+Y[i]/alphay_sum)
        t0=(1-Y[i]*Y[0])
        t1=kernel(X[0],X[0])*2*alphay_sum*Y[i]
        t2=0
        for j in range(1,len(alphas)):
            t2+=alphas[j]*Y[j]*kernel(X[j],X[0])
        t3=kernel(X[i],X[0])*alphay_sum
        t4=0
        for j in range(1,len(alphas)):
            t4+=alphas[j]*Y[j]*kernel(X[j],X[i])
        t4=t4*2*Y[i]
        gradients[i-1]=teta-t0+0.5*(t1-2*Y[i]*(t2+t3)+t4)

    new_alphas=np.array(alphas[1:])
    new_alphas=np.reshape(new_alphas,(3,1))
    new_alphas=list(new_alphas-0.01*gradients)
    var=0
    for i in range(1,len(alphas)):
        var=var+(Y[i]*new_alphas[i-1])
    var*=-1
    new_alphas.insert(0,var)
    return new_alphas

if __name__ == "__main__":
    X=[[-1,1],[-1,-1],[1,1],[1,-1]]
    Y=[1,-1,-1,1]
    alphas=[5,5,5,5]
    eta=1
    #while eta>0:
    for _ in range(250):
        alphas=newton_method(alphas,X,Y,eta)
        eta=eta/2
        print(eta, alphas[0],alphas[1],alphas[2],alphas[3])
    
        
import numpy as np
import matplotlib.pyplot as plt
import sys

def generate_data(m,k,eta):
    data=[]
    for data_no in range(m):
        X=list(np.random.normal(0,1,k-1))
        Xk=np.random.exponential(1,1)[0]+eta
        if np.random.uniform()<0.5:
            X.append(Xk)
        else:
            X.append(-Xk)
        X=[1]+X
        if X[-1]>0:
            Y=1
        else:
            Y=-1
        data.append((X,Y))
    return data



def train_perceptron(k,train_data):
    weights=np.zeros((k+1))
    weights=np.reshape(weights,(1,k+1))
    loop_count=1
    while True:
        old_weights=weights.copy()
        for data in train_data:
            X=np.reshape(data[0],(k+1,1))
            expectedY=np.matmul(weights,X)[0][0]
            if expectedY>0:
                expectedY=1
            else:
                expectedY=-1
            if expectedY!=data[1]:
               weights+=(data[1]*np.reshape(data[0],(1,k+1)))
            #   print(data[0],data[1]*np.reshape(data[0],(1,k+1)))
            #else:
            #      print("ok")
        if np.array_equal(old_weights,weights):
            return weights,loop_count
        else:
            loop_count+=1
            

def generate_data_q5(m,k):
    data=[]
    for data_no in range(m):
        X=list(np.random.normal(0,1,k))
        X2=[x**2 for x in X]
        X=[1]+X
        if np.sum(X2)>=k:
            Y=1
        else:
            Y=-1
        data.append((X,Y))
    return data


if __name__ == "__main__":
    figno=0
    
    #Q2
    m=10000
    k=20
    eta=1
    train_data = generate_data(m, k, eta)
    print(train_data)
    weights,loops=train_perceptron(k,train_data)
    print(weights)
    #print(data)
    
    
    #Q3
    etas=np.arange(0,1,0.1)
    loops_main=[]
    for eta in etas:
        loop_counts=[]
        for rep in range(20):
            train_data=generate_data(m,k,eta)
            ws,loops=train_perceptron(k,train_data)
            loop_counts.append(loops)
        loops_main.append(np.mean(loop_counts))
    plt.figure(figno)
    figno+=1
    plt.plot(etas,loops_main)
    plt.xlabel("Eta")
    plt.ylabel("Average number of steps")
    
    #Q4
    eta=1
    ks=np.arange(2,40,1)
    for m in ([100,1000]):
        loops_main=[]
        for k in ks:
            loop_counts=[]
            for rep in range(1000):
                sys.stdout.write("\r m:" +str(m)+"k: "+str(k) +"rep: "+str(rep)+"       ")
                train_data=generate_data(m,k,eta)
                ws,loops=train_perceptron(k,train_data)
                loop_counts.append(loops)
            loops_main.append(np.mean(loop_counts))
        plt.figure(figno)
        figno+=1
        plt.plot(ks,loops_main)
        plt.title("m= "+ str(m))
        plt.xlabel("k")
        plt.ylabel("Average number of steps")
        

    #Q5
    
    
        
        
        
    
    
    
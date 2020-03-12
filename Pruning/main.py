from generate_data import *
from decision_trees import *

import numpy as np
import matplotlib.pyplot as plt
import sys

global name
if __name__ == "__main__":
    m1=8000
    m2=2000
    
    """
    # Q1
    erros_main=[]
    Ms=np.arange(500,15000,1000)
    for m in Ms:
        errors=[]
        for rep in range(21):
            sys.stdout.write("\r m:"+str(m)+" rep:"+str(rep))
            train_data=generate_data(m)
            tree = decision_tree(21,train_data,[])
            
            test_data=generate_data(1000)
            errors.append(error(tree, test_data))
        erros_main.append(np.mean(errors))
    plt.plot(Ms,erros_main)
    plt.xlabel("M")
    plt.ylabel("Average_error (%)")
    """
    
    # Q2
    Ms=np.arange(10000,150000, 25000)
    irrelevant=[]
    for m in Ms:
        irrels=[]
        for rep in range(15):
            irrel=[]
            sys.stdout.write("\r m:"+str(m)+" rep:"+str(rep))
            train_data=generate_data(m)
            tree=decision_tree(21, train_data,[],s=1500)
            stack=[tree]
            while stack:
                popped=stack.pop()
                if popped[0]!="p":
                    stack.append(popped[1][0])
                    stack.append(popped[1][1])
                    if popped[0]>14:
                        irrel=set(irrel).union(set([popped[0]]))
                
            irrels.append(len(irrel))
        irrelevant.append(np.mean(irrels))
        print(irrelevant)
        print("\n")
    plt.plot(Ms,irrelevant)
    plt.xlabel("M")
    plt.ylabel("No. of irrelevant variables")
    
    """
    # Q3a
    
    errors_main=[]
    Ds=np.arange(0,20,1)
    for d in Ds:
        errors=[]
        for rep in range(20):
            sys.stdout.write("\r d: "+str(d)+" rep: "+str(rep))
            train_data=generate_data(m1)
            tree=decision_tree(21, train_data, [],d=d)
            test_data=generate_data(m2)
            errors.append(error(tree, test_data))
        errors_main.append(np.mean(errors))
        print(errors_main)
        print("\n")
    plt.plot(Ds, errors_main)
    plt.xlabel("Depth d")
    plt.ylabel("Average Error (%)")
    """
    """
    # Q3b
    
    errors_main=[]
    Ss=np.arange(0,200,25)
    for s in Ss:
        errors=[]
        for rep in range(20):
            sys.stdout.write("\r s: "+str(s)+" rep: "+str(rep))
            train_data=generate_data(m1)
            tree=decision_tree(21, train_data, [],s=s)
            test_data=generate_data(m2)
            errors.append(error(tree, test_data))
        errors_main.append(np.mean(errors))
        print(errors_main)
        print("\n")
    plt.plot(Ss, errors_main)
    plt.xlabel("Sample size (s)")
    plt.ylabel("Average Error (%)")
    """
    """
    #Q3c
    errors_main=[]
    Ts=np.arange(0,10,1)
    for T in Ts:
        errors=[]
        for rep in range(20):
            sys.stdout.write("\r T: "+str(T)+" rep: "+str(rep))
            train_data=generate_data(m1)
            tree=decision_tree(21, train_data, [],T=T)
            test_data=generate_data(m2)
            errors.append(error(tree, test_data))
        errors_main.append(np.mean(errors))
        print(errors_main)
        print("\n")
    plt.plot(Ts, errors_main)
    plt.xlabel("Significance Threshold (T)")
    plt.ylabel("Average Error (%)")
    """
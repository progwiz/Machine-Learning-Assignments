
from generate_data import *
from decision_trees import *

import numpy as np
import math
import matplotlib.pyplot as plt

if __name__=="__main__":
    k=4
    m1=100
    train_data=generate_data(k,m1)
    #print("\n\nTrain Data: \n")  
    #print(train_data)
    
    tree1, train_error=train_model(k,train_data)
    #print(tree)
    print("\n Train error:")
    print(train_error)
    
    #Q3
    m2=1000
    test_data=generate_data(k,m2)
    print("\n Q3. Test error:")
    print(error(tree1, test_data))
    """
    print("\n Decision Tree: \n")
    tree2, train_error=train_model(k,train_data,ig=False)
    print("\n Train error:")
    print(train_error)
    
    print("\n Q3. Test error:")
    print(error(tree2, test_data))
    """
    #Q4, Q5
    k=10
    m2=1000
    M=np.arange(50,500,10)
    errors1=[]
    errors2=[]
    for m1 in M:
        vals1=[]
        vals2=[]
        for rep in range(100):
            train_data=generate_data(k,m1)
            test_data=generate_data(k,m2)
            tree1, train_error=train_model(k,train_data)
            test_error=error(tree1, test_data)
            vals1.append(abs(train_error-test_error))
            
            tree2, train_error=train_model(k,train_data, ig=False)
            test_error=error(tree2, test_data)
            vals2.append(abs(train_error-test_error))
            
        errors1.append(np.mean(vals1))
        errors2.append(np.mean(vals2))
    plt.figure(0)
    plt.plot(M,errors1)
    plt.plot(M,errors2)
    plt.legend(["Information gain metric","Alternative metric"])
    plt.xlabel("M")
    plt.ylabel("|err_train-err_test|")
    
    
    
import numpy as np
import math

def decision_tree(k,data,used, ig=True):
    #calculate H(Y)
    Y0_data=[d for d in data if d[1]==0]
    
    if len(Y0_data)==0:
        return "p",[0,1]

    if len(Y0_data)==len(data):
        return "p",[1,0]
    
    if len(used)==4:   
        p0=len(Y0_data)/len(data)
        p1=1-p0
        return "p",[p0, p1]
        
    else:
        
        #calculate H(Y)
        Y0_term=len(Y0_data)/len(data)
        Y0_term*=math.log(Y0_term,2)
        Y1_term=(len(data)-len(Y0_data))/len(data)
        Y1_term+=math.log(Y1_term,2)
        H_Y=(-1)*(Y0_term+Y1_term)
            
        #calculate H(Y|X)
        current_IG=-99999
        for i in range(k):
            if i not in used:
                
                Xi0_data=[d for d in data if d[0][i]==0]
                Xi1_data=[d for d in data if d[0][i]==1]
                
                Xi0_Y0_data=[d for d in Xi0_data if d[1]==0]
                Xi1_Y0_data=[d for d in Xi0_data if d[1]==0]
                
                if ig:
                    try:
                        Xi0_Y0_term=len(Xi0_Y0_data)/len(Xi0_data)
                        Xi0_Y0_term*=math.log(Xi0_Y0_term)
                        Xi0_Y1_term=(len(Xi0_data)-len(Xi0_Y0_data))/len(Xi0_data)
                        Xi0_Y1_term*=math.log(Xi0_Y1_term)
                        Xi0_term=(-1)*(Xi0_Y0_term+Xi0_Y1_term)*(len(Xi0_data)/len(data))
                    except: 
                        Xi0_term=0
                    
                    try:
                        Xi1_Y0_term=len(Xi1_Y0_data)/len(Xi1_data)
                        Xi1_Y0_term*=math.log(Xi1_Y0_term)
                        Xi1_Y1_term=(len(Xi1_data)-len(Xi1_Y0_data))/len(Xi1_data)
                        Xi1_Y1_term*=math.log(Xi1_Y1_term)
                        Xi1_term=(-1)*(Xi1_Y0_term+Xi1_Y1_term)*(len(Xi1_data)/len(data))
                    except:
                        Xi1_term=0
                else:
                    try:
                        Xi0_term=abs(2*len(Xi0_Y0_data)-(len(Xi0_data)))*len(Xi0_data)
                    except: 
                        Xi0_term=0
                    try:
                        Xi1_term=abs(2*len(Xi1_Y0_data)-(len(Xi1_data)))*len(Xi1_data)
                    except: 
                        Xi0_term=0

                    
                    
                H_Y_Xi=Xi0_term+Xi1_term
                IG_Xi=H_Y-H_Y_Xi
                if current_IG<IG_Xi:
                    current_IG=IG_Xi
                    current_var=i
                    data0=Xi0_data
                    data1=Xi1_data
                    
        # end function
        used.append(current_var)
        branch1=decision_tree(k,data0,used.copy(),ig)
        branch2=decision_tree(k,data1,used.copy(),ig)
    
    return current_var,[branch1,branch2]
            
            
        
def error(decision_tree, test_data):
    count=0
    for data_point in test_data:
        current_var=decision_tree[0]
        current_branch=decision_tree[1]
        while current_var!="p":
            current_val=data_point[0][current_var]
            current_var=current_branch[current_val][0]
            current_branch=current_branch[current_val][1]
        if current_branch[0]==0:
            Y=1
        elif current_branch[0]==1:
            Y=0
        else:
            random_number=np.random.uniform()
            if random_number<=current_branch[0]:
                Y=0
            else:
                Y=1
        if Y!=data_point[1]:
            count+=1
    return count/len(test_data)

def train_model(k,train_data, ig=True):
    tree=decision_tree(k,train_data,[],ig)
    train_error = error(tree, train_data)
    return tree, train_error
import numpy as np
import math

def decision_tree(k,data,used, d="k",s=1, T=0):
    Y0_data=[d for d in data if d[1]==0]
    
    # all Ys are 1s or 0s    
    if len(Y0_data)==0:
        return "p",[0,1]
    if len(Y0_data)==len(data):
        return "p",[1,0]
    
    # depth for Q3a
    if d=="k":
        d=k

    if len(used)==d:   
        p0=len(Y0_data)
        if 2*p0>len(data):
            return "p",[1,0]
        else:
            return "p",[0,1]
    
    # s for Q3b
    if len(data)<s:
        p0=len(Y0_data)
        if 2*p0>len(data):
            return "p",[1,0]
        elif 2*p0<len(data):
            return "p",[0,1]
        else:
            r=np.random.uniform()
            if r<0.5:
                return "p",[1,0]
            else:
                return "p",[0,1]
    
        
    #calculate H(Y)
    Y0_term=len(Y0_data)/len(data)
    Y0_term*=math.log(Y0_term,2)
    Y1_term=(len(data)-len(Y0_data))/len(data)
    Y1_term*=math.log(Y1_term,2)
    H_Y=(-1)*(Y0_term+Y1_term)
        
    #calculate H(Y|X)
    current_IG=-99999
    for i in range(k):
        if i not in used:
            # for Q3c
            t_sum=0
            
            Xi0_data=[d for d in data if d[0][i]==0]
            Xi1_data=[d for d in data if d[0][i]==1]
            
            Xi0_Y0_data=[d for d in Xi0_data if d[1]==0]
            Xi1_Y0_data=[d for d in Xi1_data if d[1]==0]
            try:
                Xi0_Y0_term=len(Xi0_Y0_data)/len(Xi0_data)
                Xi0_Y0_term*=math.log(Xi0_Y0_term,2)
                Xi0_Y1_term=(len(Xi0_data)-len(Xi0_Y0_data))/len(Xi0_data)
                Xi0_Y1_term*=math.log(Xi0_Y1_term,2)
                Xi0_term=(-1)*(Xi0_Y0_term+Xi0_Y1_term)*(len(Xi0_data)/len(data))
                
            except: 
                Xi0_term=0
            
            try:
                Xi1_Y0_term=len(Xi1_Y0_data)/len(Xi1_data)
                Xi1_Y0_term*=math.log(Xi1_Y0_term,2)
                Xi1_Y1_term=(len(Xi1_data)-len(Xi1_Y0_data))/len(Xi1_data)
                Xi1_Y1_term*=math.log(Xi1_Y1_term,2)
                Xi1_term=(-1)*(Xi1_Y0_term+Xi1_Y1_term)*(len(Xi1_data)/len(data))
            
            except:
                Xi1_term=0
            
            
            
            H_Y_Xi=Xi0_term+Xi1_term
            IG_Xi=H_Y-H_Y_Xi
            if current_IG<IG_Xi:
                try:
                    E00=len(Xi0_data)*len(Y0_data)/len(data)
                    t_sum+=((len(Xi0_Y0_data)-E00)**2/E00)
                except: pass
                try: 
                    E10=len(Xi1_data)*len(Y0_data)/len(data)
                    t_sum+=((len(Xi1_Y0_data)-E10)**2/E10)
                except: pass
                try:
                    E01=len(Xi0_data)*(len(data)-len(Y0_data))/len(data)
                    t_sum+=(((len(Xi0_data)-len(Xi0_Y0_data))-E01)**2/E01)
                except: pass
                try: 
                    E11=len(Xi1_data)*(len(data)-len(Y0_data))/len(data)
                    t_sum+=(((len(Xi1_data)-len(Xi1_Y0_data))-E11)**2/E11)
                except: pass
                
                if t_sum>T:
                    current_IG=IG_Xi
                    current_var=i
                    data0=Xi0_data
                    data1=Xi1_data
    
    
    # terminate if all IG = 0
    if current_IG==0 or current_IG==-99999:
        #print(current_IG)
        p0=len(Y0_data)
        if 2*p0>len(data):
            return "p",[1,0]
        else:
            return "p",[0,1]
    
    
    # end function
    
    used.append(current_var)
    branch1=decision_tree(k,data0,used.copy(),d,s,T)
    branch2=decision_tree(k,data1,used.copy(),d,s,T)
    
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

def train_model(k,train_data):
    tree=decision_tree(k,train_data,[])
    train_error = error(tree, train_data)
    return tree, train_error
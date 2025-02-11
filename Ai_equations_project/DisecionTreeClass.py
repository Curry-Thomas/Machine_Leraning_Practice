import numpy as np
import pandas as pd



class DecisionTree:
    
    #Constructor
    def __init__(self,depth=0,max_depth=5):
        self.left = None
        self.right = None
        self.fkey = None
        self.fval = None
        self.max_depth = max_depth
        self.depth = depth
        self.target = None
    def entropy(self, col):
        counts = np.unique(col,return_counts=True)
        N = float(col.shape[0])
        ent = 0.0
        for ix in counts[1]:
            p  = ix/N
            ent += (-1.0*p*np.log2(p))
        return ent
    
    def divide_data(self, x_data,fkey,fval):
        '''
        x_right = pd.DataFrame([],columns=x_data.columns)
        x_left = pd.DataFrame([],columns=x_data.columns)
        
        for ix in range(x_data.shape[0]):
            val = x_data[fkey].loc[ix]
            if val > fval:
                x_right = pd.concat([x_right, x_data.loc[ix]])
            else:
                x_left = pd.concat([x_left, x_data.loc[ix]])
        '''
        x_right = x_data[x_data[fkey] > fval]
        x_left = x_data[x_data[fkey] <= fval]

        #print(x_left.shape, x_right.shape)
        return x_left,x_right

    def information_gain(self, x_data,target,fkey,fval):
    
        left,right = self.divide_data(x_data,fkey,fval)
        l = float(left.shape[0])/x_data.shape[0]
        r = float(right.shape[0])/x_data.shape[0]
        #print(l, r)

        if left.shape[0] == 0 or right.shape[0] ==0:
            return -1000000 #Min Information Gain
    
        i_gain = self.entropy(x_data[target]) - (l*self.entropy(left[target])+r*self.entropy(right[target]))
        return i_gain
        
    def train(self,X_train, target):
        
        features = X_train.columns.values
        info_gains = []
        #print(features)
        for ix in features:
            if (ix != target):
                i_gain = self.information_gain(X_train,target,ix,X_train[ix].mean())
                #print(ix,":",i_gain)
                info_gains.append(i_gain)
            
        self.fkey = features[np.argmax(info_gains)]
        self.fval = X_train[self.fkey].mean()
        #print("Making Tree Features is",self.fkey)
        
        #Split Data
        data_left,data_right = self.divide_data(X_train,self.fkey,self.fval)
        data_left = data_left.reset_index(drop=True)
        data_right = data_right.reset_index(drop=True)
         
        #Truly a left node
        if data_left.shape[0]  == 0 or data_right.shape[0] ==0:
            if X_train[target].mean() >= 0.5:
                self.target = 1
            else:
                self.target = 0
            
            return
        #Stop earyly when depth >= max depth
        if(self.depth>=self.max_depth):
            if X_train[target].mean() >= 0.5:
                self.target = 1
            else:
                self.target = 0
            return
        
        #Recursive Case
        self.left = DecisionTree(depth=self.depth+1,max_depth=self.max_depth)
        self.left.train(data_left,target)
        
        self.right = DecisionTree(depth=self.depth+1,max_depth=self.max_depth)
        self.right.train(data_right,target)
        
        #You can set the target at every node
        if X_train[target].mean() >= 0.5:
            self.target = 1
        else:
            self.target = 0
        return
    
    def predict(self,test):
        if test[self.fkey]>self.fval:
            #go to right
            if self.right is None:
                return self.target
            return self.right.predict(test)
        else:
            if self.left is None:
                return self.target
            return self.left.predict(test)

    def create_table(self):
        def store(array,node, row):
            col = node.depth*2
            if (node.left is None):
                array[col][row] = "Target"
                array[col+1][row] = node.target
                row += 1
            else:
                array[col][row] = node.fkey
                array[col+1][row] = node.fval
                array, row = store(array, node.left, row)
                array, row = store(array, node.right, row)
            return array, row
        storage = [[None for _ in range(2 ** self.max_depth)] for _ in range((self.max_depth+1)*2)] 
        storage, r = store(storage, self, 0)

        return pd.DataFrame(storage)
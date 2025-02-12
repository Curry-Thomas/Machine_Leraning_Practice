import numpy as np

class FNN: 
    def __init__(self): 
        self.parameters = {} 

    def standardize(self, train_input):
        train_input_st = np.copy(train_input)
        for i in range(train_input_st.shape[1]):
            train_input_st[:,i] = (train_input_st[:,i] - np.mean(train_input_st[:,i]))/np.std(train_input_st[:,i])
        return train_input_st
    
    def sigmoid(self,z):
        clip_z = np.clip(z,-6,6)
        return np.divide(1, 1 + np.exp( -1 * clip_z))
    
    def cross_etropy(self, predictions, true_out):
        positive = np.divide(true_out,predictions)
        negitive = np.divide(1-true_out, 1- predictions) 
        return negitive - positive
    
    def cross_entropy_cost(self, predictions, true_out):
        positive = np.multiply(true_out,np.log(predictions))
        negitive = np.multiply(1-true_out,np.log(1-predictions))
        entropy = positive + negitive
        return -1 * np.mean(entropy)
    def update_parameters(self, derivatives, learning_rate): 
        self.parameters['W1'] = self.parameters['W1'] - learning_rate * derivatives['dW1']
        self.parameters['W2'] = self.parameters['W2'] - learning_rate * derivatives['dW2']
        self.parameters['b1'] = self.parameters['b1'] - learning_rate * derivatives['db1']
        self.parameters['b2'] = self.parameters['b2'] - learning_rate * derivatives['db2']
        #self.parameters['c'] = self.parameters['c'] - learning_rate * derivatives['dc'] 

    def train(self, train_input, train_output, learning_rate, iters,batch_size = 25): 
        # Initialize random parameters 
        self.parameters['W1'] = np.random.uniform(-1, 1, (train_input.shape[1], 10))
        self.parameters['W2'] = np.random.uniform(-1, 1, (10, 1)) 
        self.parameters["b1"] = np.random.uniform(-1,1,(1,10))
        self.parameters["b2"] = np.random.uniform(-1,1,(1,1))
        #self.parameters['c'] = np.random.uniform(0, 1) 
        derivatives = {} 
        
        #oppp_out = np.absolute((train_output - np.ones(train_output.shape)))
        #train_output_2D = np.concatenate((train_output, oppp_out), axis=1)
        
        train_input_st = self.standardize(train_input)
        
        #print(train_output_2D.shape)
        # Initialize loss 
        self.loss = [] 
        batch_n = len(train_output) / batch_size
        print(batch_n)

        def update(frame): 
            # Forward propagation 
            Z = np.dot(train_input_st, self.parameters['W1']) + self.parameters["b1"]
            # Cost function 
            A = self.sigmoid(Z)
            y= np.dot(A, self.parameters['W2']) + self.parameters["b2"]
            predictions = self.sigmoid(y)
            #cost = self.cost_function(predictions, train_output)
            # Back propagation 
            error = self.cross_etropy(predictions,train_output)
            dri_pred = np.multiply(predictions, (1-predictions))
            dri_A = np.multiply(A, (1-A))
            dri_error = np.multiply(dri_pred, error)
            dW1 = np.zeros(self.parameters['W1'].shape)
            dW2 = np.zeros(self.parameters['W2'].shape)
            db2 = np.zeros(self.parameters['b2'].shape)
            db1 = np.zeros(self.parameters['b1'].shape)
            if (frame % 10 == 0):
                print(self.cross_entropy_cost(predictions,train_output)) #, np.mean(predictions,axis=0))
                print(frame,self.test_metritcs(predictions, train_output))
                print(np.median(dri_error))
            for i in range(A.shape[0]):
                dW2 = dW2 + np.multiply(A[i].reshape(10,1), dri_error[i].reshape(1,1))
                db2 = db2 + dri_error[i].reshape(1,1)
                s1 = np.dot(dri_error[i],self.parameters['W2'].T)
                s2 = np.multiply(s1, dri_A[i])
                dW1 = dW1 + np.multiply(s2,train_input_st[i].reshape(13,1))
                db1 = db1 +  s2
                if(((i+1)%batch_n) == 0):
                    derivatives['dW2'] = dW2 / batch_n
                    derivatives['dW1'] = dW1 / batch_n
                    derivatives['db2'] = db2 / batch_n
                    derivatives['db1'] = db1 / batch_n
                    self.update_parameters(derivatives, learning_rate)
                    dW1 = np.zeros(self.parameters['W1'].shape)
                    dW2 = np.zeros(self.parameters['W2'].shape)
                    db2 = np.zeros(self.parameters['b2'].shape)
                    db1 = np.zeros(self.parameters['b1'].shape)
                #print(derivatives)
            '''    
                if(((i+1)%100) == 0):
                    derivatives['dW2'] = derivatives['dW2'] / 10
                    derivatives['dW1'] = derivatives['dW1'] / 10
                    
                    derivatives['dW1'] =np.zeros((train_input.shape[1], 10))
                    derivatives['dW2'] =np.zeros((10,1))

                #print(derivatives)
            # Update parameters 
          
            derivatives['dW2'] = np.dot(A.T, error)
            derivatives['dW1'] = 
            
            # Append loss and print
            cost = self.cost_function(predictions, train_output)[0]
           
            
            
            self.loss.append(cost) 
            
            #return line,
        '''
        for i in range(iters):update(i)       

        return self.parameters, self.loss 
   
    
    
    def test_metritcs(self, predictions, test_output):
        results = [0,0,0,0]
        mean = np.mean(test_output)
        for i in range(len(predictions)):
            if (predictions[i] >= mean and test_output[i] == 1):
                results[0] += 1
            elif (predictions[i] >= mean and test_output[i] == 0):
                results[1] += 1
            elif (predictions[i] < mean and test_output[i] == 0):
                results[2] += 1
            elif (predictions[i] < mean and test_output[i] == 1):
                results[3] += 1
        print(results)
        if ((results[0] + results[1]) != 0):
            Precision = results[0] / (results[0] + results[1])
        else: Precision = -1
        if ((results[0] + results[3]) != 0):
            recall = results[0] / (results[0] + results[3])
        else: recall = -1
        if ((results[2] + results[3]) != 0):
            PrecisionN = results[2] / (results[2] + results[3])
        else: PrecisionN = -1
        if ((results[1] + results[2]) != 0):
            recallN = results[2] / (results[1] + results[2])
        else: recallN = -1
        Acc = (results[0] + results[2])/sum(results)
        return [Precision, recall, PrecisionN, recallN, Acc]

    def use(self, data):
        data_st = self.standardize(data)

        Z = np.dot(data_st, self.parameters['W1']) + self.parameters["b1"]
            # Cost function 
        A = self.sigmoid(Z)
        y= np.dot(A, self.parameters['W2']) + self.parameters["b2"]
        predictions = self.sigmoid(y)
        return predictions

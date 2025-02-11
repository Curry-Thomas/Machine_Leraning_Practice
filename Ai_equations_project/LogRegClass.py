import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from matplotlib.animation import FuncAnimation

class LogisticRegression: 
    def __init__(self): 
        self.parameters = {} 

    def find_z(self, train_input): 
        m = self.parameters['m'] 
        c = self.parameters['c'] 
        #print(m.shape, train_input.shape)
        predict = np.dot(train_input, m) + c
        return np.array(predict).reshape(len(predict))

    def cost_function(self, predictions, train_output): 
        logSig = np.log(predictions) 
        InvlogSig = np.log(1-predictions)
        cost_array = 0
        for i in range(len(predictions)):
            cost_array += train_output[i] * logSig[i] + (1-train_output[i]) * InvlogSig[i] 
        return -1 * cost_array / len(predictions)
    
    def standardize(self, train_input):
        train_input_st = np.copy(train_input)
        for i in range(train_input_st.shape[1]):
            train_input_st[:,i] = (train_input_st[:,i] - np.mean(train_input_st[:,i]))/np.std(train_input_st[:,i])
        return train_input_st

    
    def sigmoid(self,z):
        zclip = np.clip(z, -5,5)
        return (np.divide(1, 1 + np.exp(-1 * zclip)))



    def backward_propagation(self, train_input, train_output, predictions): 
        derivatives = {} 
        #print(train_output.shape,predictions.shape)
        df = (predictions-train_output.reshape(len(train_output))).reshape(len(predictions), 1)
        dm = np.divide(np.dot(train_input.T , df),  len(predictions))
        #print(dm.shape)
        dc = np.mean(df) 
        derivatives['dm'] = dm.reshape(len(dm))
        derivatives['dc'] = dc 
        return derivatives 

    def update_parameters(self, derivatives, learning_rate): 
        self.parameters['m'] = self.parameters['m'] - learning_rate * derivatives['dm']
        self.parameters['c'] = self.parameters['c'] - learning_rate * derivatives['dc'] 

    def train(self, train_input, train_output, learning_rate, iters, Show_Acc = False): 
        # Initialize random parameters 
        self.parameters['m'] = np.random.uniform(0, 1, train_input.shape[1]) 
        self.parameters['c'] = np.random.uniform(0, 1) 
        
        train_input_st = self.standardize(train_input)
        
        # Initialize loss 
        self.loss = [] 
        def update(frame): 
            # Forward propagation 
            Z = self.find_z(train_input_st) 
            # Cost function 
            predictions = self.sigmoid(Z)
            #cost = self.cost_function(predictions, train_output)
            # Back propagation 
            derivatives = self.backward_propagation(train_input_st, train_output, predictions) 
            # Update parameters 
            self.update_parameters(derivatives, learning_rate) 
            # Append loss and print
            cost = self.cost_function(predictions, train_output)[0]
            self.loss.append(cost) 
            if (Show_Acc):
                if((frame % 10) == 0):
                    prec, recal,precN, recalN, acur =self.test_metritcs(predictions, train_output)
                    print("Iteration = {}, Precision = {}, Recall = {}".format(frame + 1, prec, recal))
                    print(" PrecisionN = {}, RecallN = {}".format(precN, recalN))
                    print("accuracy =",acur)
                    print("Loss: ",cost )
            #return line,
        
        for i in range(iters):update(i)

        # Create animation 
       

        return self.parameters, self.loss 
    
    def Ani_train(self, train_input, train_output, learning_rate, iters): 
        # Initialize random parameters 
        self.parameters['m'] = np.random.uniform(0, 1, train_input.shape[1]) 
        self.parameters['c'] = np.random.uniform(0, 1) 
        # Initialize loss 
        self.loss = []
        
        train_input_st = self.standardize(train_input)
        fig, ax = plt.subplots() 
        x_vals = np.linspace(-20,20, 1000) 
        ax.plot(x_vals, self.sigmoid(x_vals), color='red', label='sigmoid Line') 
        sct, = ax.plot(train_output.reshape(len(train_output)),train_output.reshape(len(train_output)), 'go') 

        def update(frame): 
            # Forward propagation 
            Z = self.find_z(train_input_st) 
            # Cost function 
            predictions = self.sigmoid(Z)
            #cost = self.cost_function(predictions, train_output)
            # Back propagation 
            derivatives = self.backward_propagation(train_input_st, train_output, predictions) 
            # Update parameters 
            self.update_parameters(derivatives, learning_rate)
            sct.set_data(Z.reshape(len(Z)),predictions.reshape(len(predictions)))
            #print(np.max(Z))
            #print(sct.get_data)

            # Append loss and print 

            if ((frame % 100) == 0):
                cost = self.cost_function(predictions, train_output)[0]
                self.loss.append(cost) 
                prec, recal,precN, recalN, acur =self.test_metritcs(predictions, train_output)
                print("Iteration = {}, Precision = {}, Recall = {}, accuracy = {}".format(frame + 1, prec, recal, acur))
                print(" PrecisionN = {}, Recall = {}".format(precN, recalN))

                print("Loss:", cost)
            return sct,
        
        #for i in range(iters):update(i)
        # Create animation 
        ani = FuncAnimation(fig, update, frames=iters, interval=100, blit=True, repeat_delay= 5000) 

        # Save the animation as a video file (e.g., MP4) 
        ani.save('log_regression_A3.gif', writer='ffmpeg') 

        plt.xlabel('Z') 
        plt.ylabel('Pedictions') 
        plt.title('log Regression') 
        plt.legend() 
        plt.show() 
        return self.parameters, self.loss 
    
    
    
    def test_metritcs(self, predictions, test_output):
        results = [0,0,0,0]
        for i in range(len(predictions)):
            if (predictions[i] >= 0.5 and test_output[i] == 1):
                results[0] += 1
            elif (predictions[i] >= 0.5 and test_output[i] == 0):
                results[1] += 1
            elif (predictions[i] < 0.5 and test_output[i] == 0):
                results[2] += 1
            elif (predictions[i] < 0.5 and test_output[i] == 1):
                results[3] += 1
        if ((results[0] + results[1]) != 0):
            Precision = results[0] / (results[0] + results[1])
        else: Precision = -1
        if ((results[0] + results[3]) != 0):
            recall = results[0] / (results[0] + results[3])
        else: recall = -1
        if ((results[2] + results[3]) != 0):
            PrecisionN = results[2] / (results[2] + results[3])
        else: Precision = -1
        if ((results[1] + results[2]) != 0):
            recallN = results[2] / (results[1] + results[2])
        else: recall = -1
        Acc = (results[0] + results[3])/sum(results)
        return Precision, recall,PrecisionN,recallN, Acc
        


    def use(self, input):
        input_st = self.standardize(input)
        out = np.dot(input_st, self.parameters["m"]) + self.parameters["c"]
        return self.sigmoid(out)
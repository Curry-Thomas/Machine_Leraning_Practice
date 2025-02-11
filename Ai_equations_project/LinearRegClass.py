import numpy as np

class LinearRegression: 
    def __init__(self): 
        self.parameters = {} 

    def forward_propagation(self, train_input): 
        m = self.parameters['m'] 
        c = self.parameters['c'] 

        predict =[]
        for rows in train_input:
            predict.append(sum(np.multiply(m, rows) + c))
        return np.array(predict).reshape(len(predict), 1)

    def cost_function(self, predictions, train_output): 
        cost = np.mean((train_output - predictions) ** 2) 
        return cost 

    def backward_propagation(self, train_input, train_output, predictions): 
        derivatives = {} 
        df = (predictions-train_output).reshape(len(predictions))

        # dm= 2/n * mean of (predictions-actual) * input 
        dm_t = []
        #print(np.transpose(train_input).shape, df.shape)
        for i in range(train_input.shape[1]):
        #    print(np.transpose(train_input)[i].shape, i)
            dm_t.append(2 * np.mean(np.multiply(np.transpose(train_input)[i], df)) )
        dm = np.array(dm_t).reshape(len(dm_t))
        # dc = 2/n * mean of (predictions-actual) 
        dc = 2 * np.mean(df) 
        derivatives['dm'] = dm 
        derivatives['dc'] = dc 
        return derivatives 

    def update_parameters(self, derivatives, learning_rate): 
        self.parameters['m'] = self.parameters['m']- learning_rate * derivatives['dm']
        self.parameters['c'] = self.parameters['c'] - learning_rate * derivatives['dc'] 

    def train(self, train_input, train_output, learning_rate, iters): 
        # Initialize random parameters 
        self.parameters['m'] = np.random.uniform(0, 1, train_input.shape[1]) 
        self.parameters['c'] = np.random.uniform(0, 1) 

        # Initialize loss 
        self.loss = [] 
      

        def update(frame): 
            # Forward propagation 
            predictions = self.forward_propagation(train_input) 
            # Cost function 
            cost = self.cost_function(predictions, train_output)
            # Back propagation 
            derivatives = self.backward_propagation(train_input, train_output, predictions) 
            # Update parameters 
            self.update_parameters(derivatives, learning_rate) 

            # Append loss and print 
            self.loss.append(cost) 
            print("Iteration = {}, Loss = {}".format(frame + 1, cost))
            #return line,
        
        for i in range(iters):update(i)
        # Create animation 


        return self.parameters, self.loss 
    
    def use(self, input):
        out = self.parameters['c']
        for i in range(input.size):
            out += self.parameters['m'][i] * input[i]
        return out
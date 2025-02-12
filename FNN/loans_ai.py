from FNN_logClass_cross_13_10_1 import FNN

import pandas as pd
import numpy as np

data = pd.read_csv("loan_data_p.csv")

#print(data.shape)
training = np.array(data[:36000])

training_in = np.array(training[:,:13])
training_out = np.array(training[:,13]).reshape(36000,1)
print(np.mean(training_out))
#print(training_in.shape, training_out.shape)
'''
testing = np.array(data[36000:])
testing_in = np.array(testing[:,:13])
testing_out = np.array(testing[:,13]).reshape(9000,1)
'''
#print(testing_in.shape, testing_out.shape)

fnn_pract = FNN()
parameters, loss = fnn_pract.train(training_in, training_out, 0.001, 301, 50)
#pd.DataFrame(fnn_pract.use(training_in)).to_csv('cross_outs_i500_b25.csv', index=False)
#pd.DataFrame(parameters["W1"]).to_csv('cross_outs_i500_b25_W1.csv', index=False,header = False)
#pd.DataFrame(parameters["W2"]).to_csv('cross_outs_i500_b25_W2.csv', index=False,header = False)

#inp44 = np.array([25,0,2,90785,4,1,30000,5,16.89,0.33,4,649,0])
#print(linear_reg.use(inp44))
#
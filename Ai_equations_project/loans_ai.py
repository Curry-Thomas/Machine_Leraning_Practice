from LogRegClass import LogisticRegression as lr

import pandas as pd
import numpy as np

data = pd.read_csv("loan_data_p.csv")

#print(data.shape)
training = np.array(data[:36000])

training_in = np.array(training[:,:13])
training_out = np.array(training[:,13]).reshape(36000,1)
#print(training_in.shape, training_out.shape)
'''
testing = np.array(data[36000:])
testing_in = np.array(testing[:,:13])
testing_out = np.array(testing[:,13]).reshape(9000,1)
'''
#print(testing_in.shape, testing_out.shape)

linear_reg = lr()
parameters, loss = linear_reg.train(training_in, training_out, 0.0001, 100, True)
#pd.DataFrame(parameters).to_csv('outLoanlog6.csv', index=False)
#inp44 = np.array([25,0,2,90785,4,1,30000,5,16.89,0.33,4,649,0])
#print(linear_reg.use(inp44))

from DisecionTreeClass import DecisionTree as dt

import pandas as pd
import numpy as np

data = pd.read_csv("loan_data_p.csv")

#print(data.shape)
training = data[:36000]
#print(training.loc[0])

#print(training_in.shape, training_out.shape)
'''
testing = np.array(data[36000:])
testing_in = np.array(testing[:,:13])
testing_out = np.array(testing[:,13]).reshape(9000,1)
'''
#print(testing_in.shape, testing_out.shape)

Decision_Tree = dt(max_depth=10)
Decision_Tree.train(training, "loan_status")
tree = Decision_Tree.create_table()
#print(tree)
pd.DataFrame(tree).to_csv('tree10.csv', index=False)

#print(Decision_Tree.predict(training.loc[0]))
''''
true_stat = training["loan_status"].sum()
print(true_stat)
perdsum = []
acc = [0,0,0,0]
for i in range(training.shape[0]):
    tem = Decision_Tree.predict(training.loc[i])
    perdsum.append(tem)
    if tem == training["loan_status"][i]:
        if tem == 1:
            acc[0] += 1
        else:
            acc[2] += 1
    if tem != training["loan_status"][i]:
        if tem == 1:
            acc[1] += 1
        else:
            acc[3] += 1
    
correct =training[perdsum == training["loan_status"]]
print(correct.shape[0]/ training.shape[0])
print(acc, )
print(acc[2]/ sum(acc[1:]))
print(acc[2]/(acc[1]+ acc[2]))


testing = data[36000:]

true_stat = testing["loan_status"].sum()
print(true_stat)
perdsum = []
acc = [0,0,0,0]
for i in range(testing.shape[0]):
    n = i + 36000
    tem = Decision_Tree.predict(testing.loc[n])
    perdsum.append(tem)
    if tem == testing["loan_status"][n]:
        if tem == 1:
            acc[0] += 1
        else:
            acc[2] += 1
    if tem != testing["loan_status"][n]:
        if tem == 1:
            acc[1] += 1
        else:
            acc[3] += 1
    
correct =testing[perdsum == testing["loan_status"]]
print(correct.shape[0]/ testing.shape[0])
print(acc, )
print(acc[2]/ sum(acc[1:]))
print(acc[2]/(acc[1]+ acc[2]))
'''
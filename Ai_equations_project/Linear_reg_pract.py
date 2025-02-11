import pandas as pd
import numpy as np
from LinearRegClass import LinearRegression as lr
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from matplotlib.animation import FuncAnimation


data = pd.read_csv("data_for_lr.csv")


# Drop the missing values
data = data.dropna()

# training dataset and labels
train_input = np.array(data.x[0:500]).reshape(500, 1)
train_output = np.array(data.y[0:500]).reshape(500, 1)
#print(train_input.shape[1])
# valid dataset and labels
test_input = np.array(data.x[500:700]).reshape(199, 1)
test_output = np.array(data.y[500:700]).reshape(199, 1)


linear_reg = lr()
parameters, loss = linear_reg.train(train_input, train_output, 0.0001, 20)


#pd.DataFrame(parameters).to_csv('out.csv', index=False)

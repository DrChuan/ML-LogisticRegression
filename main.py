import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression

# read in the dataset and pre-process it
data = pd.read_csv('diabetes.csv')

# split
shuffle_index = np.random.permutation(768)
data_sh = data.values[shuffle_index]
X_train, y_train = data_sh[:500, :-1], data_sh[:500, -1]
X_test, y_test = data_sh[500:, :-1], data_sh[500:, -1]
y_train = y_train.astype('int64')
y_test = y_test.astype('int64')

# normalization
mean = X_train.mean(0)
std = X_train.std(0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# build and train the model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train, 0.001, 100)

# test for precision and recall
rg = np.arange(0.1, 0.9, 0.05)
precisions = []
recalls = []

for i in rg:
    log_reg.predict(X_test, y_test, i)
    precisions.append(log_reg.precision)
    recalls.append(log_reg.recall)

# plot the graph
plt.plot(rg, precisions)
plt.plot(rg, recalls)
plt.show()

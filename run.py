from neuralnetwork import *
import pandas as pd
import numpy as np

network = NeuralNetwork()

X = pd.read_csv('X.csv')
Y = pd.read_csv('Y.csv')

all_y_trues = pd.read_csv('Y.csv')
all_y_trues = pd.to_numeric(all_y_trues['survived'])

data = pd.read_csv('X.csv').drop(columns=['name','cabin','fare','ticket']).replace('male',0).replace('female',1).replace('S',3).replace('C',2).replace('Q',1).replace(np.nan,45)
cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'embarked']
for col in cols:  # Iterate over chosen columns
	data[col] = pd.to_numeric(data[col],downcast='float')


data 		= np.array_split(data, 8)[0]
all_y_trues = np.array_split(all_y_trues, 8)[0]



network.train(data,all_y_trues)

emily = np.array([1,1,10,1,0,3]) # 128 pounds, 63 inches
frank = np.array([2,0,45,1,0,2])  # 155 pounds, 68 inches

print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - M
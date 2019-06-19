import numpy as np
from utils import *

def intt(n):
    z = np.zeros((n))
    for zz in z:
        zz = np.random.normal()
    return z

class NeuralNetwork:
    def __init__(self):
        
        self.wh1 = intt(6)
        self.wh2 = intt(6)
        self.wh3 = intt(6)

        self.wh4 = intt(3)
        self.wh5 = intt(3)

        self.wo1 = intt(2)

        self.bh1 = np.random.normal()
        self.bh2 = np.random.normal()
        self.bh3 = np.random.normal()

        self.bh4 = np.random.normal()
        self.bh5 = np.random.normal()

        self.bo1 = np.random.normal()

    def config(self,data,all_y_trues):
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        loss = mse_loss(all_y_trues, y_preds)
        print("loss: %.9f" % loss)



    def feedforward(self,x):
        h1 = sigmoid(np.dot(self.wh1, x) + self.bh1)
        h2 = sigmoid(np.dot(self.wh2, x) + self.bh2)
        h3 = sigmoid(np.dot(self.wh3, x) + self.bh3)

        h4 = sigmoid(np.dot(self.wh4, [h1,h2,h3]) + self.bh4)
        h5 = sigmoid(np.dot(self.wh5, [h1,h2,h3]) + self.bh5)

        o1 = sigmoid(np.dot(self.wo1, [h4,h5]) + self.bo1)

        return o1

    def train(self,data,all_y_trues):
        '''
        - data is a (n x 2) numpy array, n = # of samples in the dataset.
        - all_y_trues is a numpy array with n elements.
        Elements in all_y_trues correspond to those in data.
        '''
        learn_rate = 0.1
        epochs = 10000 # number of times to loop through the entire dataset

        for epoch in range(epochs):
            for i in range(data.count().embarked):
                x = data.loc[i]
                y_true = all_y_trues.loc[i]
        # --- Do a feedforward (we'll need these values later)
                sum_h1 = np.dot(self.wh1, x) + self.bh1
                sum_h2 = np.dot(self.wh2, x) + self.bh2
                sum_h3 = np.dot(self.wh3, x) + self.bh3

                h1 = sigmoid(sum_h1)
                h2 = sigmoid(sum_h2)
                h3 = sigmoid(sum_h3)

                sum_h4 = np.dot(self.wh4, [h1,h2,h3]) + self.bh4
                sum_h5 = np.dot(self.wh5, [h1,h2,h3]) + self.bh5

                h4 = sigmoid(sum_h4)
                h5 = sigmoid(sum_h5)

                sum_o1 = np.dot(self.wo1, [h4,h5]) + self.bo1

                o1 = sigmoid(sum_o1)

                y_pred = o1

                d_L_d_ypred = -2 * (y_true - y_pred)

                d_w_for_h1          = np.zeros((6))
                d_w_for_h2          = np.zeros((6))
                d_w_for_h3          = np.zeros((6))

                d_w_for_h4          = np.zeros((3))
                d_w_for_h5          = np.zeros((3))

                d_w_for_o1          = np.zeros((2))

                base = learn_rate * d_L_d_ypred

                #DW1-DW6

                for i in range(6):
                    d_w_for_h1[i] = dw_1_18(sum_h1,sum_h4,sum_h5,sum_o1,self.wo1,self.wh4[0],self.wh5[0],x[i])

                #DW7 - DW12

                for i in range(6):
                    d_w_for_h2[i] = dw_1_18(sum_h2,sum_h4,sum_h5,sum_o1,self.wo1,self.wh4[1],self.wh5[1],x[i])

                #DW13-DW18

                for i in range(6):
                    d_w_for_h3[i] = dw_1_18(sum_h3,sum_h4,sum_h5,sum_o1,self.wo1,self.wh4[2],self.wh5[2],x[i])

                #DW19-DW24

                d_w_for_h4[0]        = dw_19_24(sum_h1,sum_h4,sum_o1,self.wo1[0])
                d_w_for_h4[1]        = dw_19_24(sum_h2,sum_h4,sum_o1,self.wo1[0])
                d_w_for_h4[2]        = dw_19_24(sum_h3,sum_h4,sum_o1,self.wo1[0])

                d_w_for_h5[0]        = dw_19_24(sum_h1,sum_h5,sum_o1,self.wo1[1])
                d_w_for_h5[1]        = dw_19_24(sum_h2,sum_h5,sum_o1,self.wo1[1])
                d_w_for_h5[2]        = dw_19_24(sum_h3,sum_h5,sum_o1,self.wo1[1])

                #DW25-DW26

                d_w_for_o1[0]        =   dw_25_26(sum_h4,sum_o1)
                d_w_for_o1[1]        =   dw_25_26(sum_h5,sum_o1)


                #DD1-DB3

                d_b_for_h1 = db_1_3(sum_h1,sum_h4,sum_h5,sum_o1,self.wo1,self.wh4[0],self.wh5[0])
                d_b_for_h2 = db_1_3(sum_h2,sum_h4,sum_h5,sum_o1,self.wo1,self.wh4[1],self.wh5[1])
                d_b_for_h3 = db_1_3(sum_h3,sum_h4,sum_h5,sum_o1,self.wo1,self.wh4[2],self.wh5[2])

                #DB4-Db5

                d_b_for_h4 = db_4_5(sum_h4,sum_o1,self.wo1[0])
                d_b_for_h5 = db_4_5(sum_h5,sum_o1,self.wo1[1])

                #DB6

                d_b_for_o1 = db_6(sum_o1)

                ##UPDATE

                #neuron h1

                for i in range(6):
                    self.wh1[i]     -= base * d_w_for_h1[i]
                
                self.bh1            -= base * d_b_for_h1
            
                #neuron h2

                for i in range(6):
                    self.wh2[i]     -= base * d_w_for_h2[i]

                self.bh2            -= base * d_b_for_h2 

                #neuron h3

                for i in range(6):
                    self.wh3[i]     -= base * d_w_for_h3[i]

                self.bh3            -= base * d_b_for_h3

                #neuron h4

                for i in range(3):
                    self.wh4[i]     -= base * d_w_for_h4[i]
                    
                self.bh4            -= base * d_b_for_h4

                #neuron h5

                for i in range(3):
                    self.wh5[i]     -= base * d_w_for_h5[i]
                
                self.bh5            -= base * d_b_for_h5

                #neuron o1

                self.wo1[0]         -= base * d_w_for_o1[0]
                self.wo1[1]         -= base * d_w_for_o1[1]

                self.bo1            -= base * d_b_for_o1

                # --- Calculate total loss at the end of each epoch
                if epoch % 10 == 0:
                    self.config(data,all_y_trues)

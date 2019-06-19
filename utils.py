import numpy as np

def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
  return ((y_true - y_pred) ** 2).mean()


def sigmoid(x):
	return 1/(1 + np.exp(-x))

def deriv_sigmoid(x):
	return sigmoid(x)*(1 - sigmoid(x))

def tanh(x):
	return 2*sigmoid(2*x) - 1

def dtanh(x):
	return 4/(np.exp(x)+np.exp(-x))**2

def dw_1_18(sum_h,sum_h4,sum_h5,sum_o,wo1,wh4,wh5,x):

	# d_ypred_dw = d_ypred_d_hl1 * d_h4_d_w1 + d_ypred_d_hl2 * d_h5_d_w1

    d_ypred_d_l1    = wo1[0]*dtanh(sum_o) #1
    d_ypred_d_l2    = wo1[1]*dtanh(sum_o)

    d_l1_d_h       = wh4*dtanh(sum_h4) #2
    d_l2_d_h       = wh5*dtanh(sum_h5)

    #common
    d_h_d_w       = x*dtanh(sum_h) #3

    ##RESULT
    d_l1_d_w = d_l1_d_h * d_h_d_w # 2 x 3
    d_l2_d_w = d_l2_d_h * d_h_d_w

    res = d_ypred_d_l1 * d_l1_d_w + d_ypred_d_l2 * d_l2_d_w
    # print(res)
    return res

def dw_19_24(sum_h,sum_l,sum_o,w):

	# d_ypred_d_w = d_ypred_d_l * d_l_d_w

    d_ypred_d_l    = w*dtanh(sum_o)
    d_l_d_w       = tanh(sum_h)*dtanh(sum_l)

    ##RESULT
    res = d_ypred_d_l * d_l_d_w
    # print('x ',res)
    return d_ypred_d_l * d_l_d_w

def dw_25_26(sum_l,sum_o):

	# d_ypred_d_w = l * d_ypred_d_w
    res = tanh(sum_l)*dtanh(sum_o)
    # print('o ',res)
    return res

def db_1_3(sum_h,sum_l1,sum_l2,sum_o,wo1,wh4,wh5):
    #DW1 - DW6

    d_ypred_d_l1    = wo1[0]*dtanh(sum_o) #1
    d_ypred_d_l2    = wo1[1]*dtanh(sum_o)

    #DW1
    # d_ypred_d_b = d_ypred_d_l1 * d_l1_d_b + d_ypred_d_l2 * d_l2_d_b

    d_l1_d_h       = wh4*dtanh(sum_l1) #2
    d_l2_d_h       = wh5*dtanh(sum_l2)

    #common
    d_h_d_b       = dtanh(sum_h)

    ##RESULT
    d_l1_d_w = d_l1_d_h * d_h_d_b
    d_l2_d_w = d_l2_d_h * d_h_d_b

    return d_ypred_d_l1 * d_l1_d_w + d_ypred_d_l2 * d_l2_d_w

def db_4_5(sum_l,sum_o,w):

	# d_ypred_d_b = d_ypred_d_l * d_l_d_b

    d_ypred_d_l    	= w*dtanh(sum_o)
    d_l_d_b       	= dtanh(sum_l)

    ##RESULT
    return d_ypred_d_l * d_l_d_b

def db_6(sum_o):

	# d_ypred_d_w = d_ypred_d_w
    
    return dtanh(sum_o)
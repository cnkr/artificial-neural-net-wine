#!/usr/bin/python3

import numpy as np
import pandas as pd
p = print
mat = np.expand_dims

def softmax(vector):
    # Last layer's activation function. Turns outputs to probability [0,1]
    
    stables = vector - np.max(vector)
    exps = np.exp(stables)
    softs = exps / np.sum(exps)
    return softs

def f_deriv(tanh_x):
    # Takes derivative inner layer activation function
    
    return (1 - np.power(tanh_x, 2))


#==============================================
# Data & Metaparams Initialization
#==============================================

df = pd.read_csv('Wines.csv')

X = df.values[:,:-3] # (178,13) features
y = df.values[:,-3:] # (178,3) outputs

# Number of neurons per level
nn_input = 13
nn_hid1 = 5
nn_hid2 = 5
nn_output = 3
learning_rate = 5.07

#np.random.seed(0)
np.set_printoptions(precision=5)

a0 = a1 = a2 = a3 = W1 = W2 = W3 = b1 = b2 = b3 = z1 = z2 = z3 = None

#==============================================
# Weights & Bias Initialization
#==============================================

# (5,13) weights of level 0->1 # (5,) bias of level 0->1
W1 = 2 * np.random.randn(nn_hid1, nn_input) - 1  
b1 = np.zeros((nn_hid1,))  
 
# (5,5) weights of level 1->2 # (5,) bias of level 2->2
W2 = 2 * np.random.randn(nn_hid2, nn_hid1) - 1  
b2 = np.zeros((nn_hid2,))

# (3,5) weights of level 2->3 # (3,) bias of level 2->3
W3 = 2 * np.random.randn(nn_output, nn_hid2) - 1  
b3 = np.zeros((nn_output,))

for epoch in range(0, 5000):
    
#    learning_rate -= learning_rate/1000

    # These will hold sums of gradients (and finally we'll take average)
    gW1 = np.zeros_like(W1)
    gW2 = np.zeros_like(W2)
    gW3 = np.zeros_like(W3)
    gb1 = np.zeros_like(b1)
    gb2 = np.zeros_like(b2)
    gb3 = np.zeros_like(b3)
    train_ctr = 0
    train_correct = 0
    train_cost = 0
    test_correct = 0
    test_ctr = 0
    test_cost = 0
    test_period = 3 # obs % period will be test
    
    # For every observation
    for obs in range(X.shape[0]):
        
        #==============================================
        # Forward Propagation for this observation
        #==============================================
        
        # (13,) input neurons (level 0 output)
        # (5,) input to first hidden layer (level 1 input)
        a0 = X[obs, :]
        z1 = W1.dot(a0) + b1  
        
        # (5,) output of first hidden layer (level 1 output)
        # (5,) input to second hidden layer (level 2 input)
        a1 = np.tanh(z1)  
        z2 = W2.dot(a1) + b2
        
        # (5,) output of second hidden layer (level 2 output)
        # (3,) input to the output layer (level 3 input)
        a2 = np.tanh(z2)  
        z3 = W3.dot(a2) + b3
        
        # (3,) NN output
        a3 = softmax(z3)
        
        #==============================================
        # Backward Propagation for this observation
        #==============================================
        
        # (3,) observed (real) output
        y_obs = y[obs,:]
        
        C = 1/2 * np.sum((y_obs - a3) ** 2)  # cost for this observation
        
        if obs % test_period == 0:
            test_cost += C
            test_ctr += 1
            if np.argmax(y_obs) == np.argmax(a3):
                    test_correct += 1
            continue
        else:
            train_cost += C
            train_ctr += 1
            if np.argmax(y_obs) == np.argmax(a3):
                train_correct += 1
        
        #-------------------------------------
        # Chained derivatives: m_n means dm/dn
        #-------------------------------------
        
        C_a3 = a3 - y_obs # 1D per output neuron (3,)
        
        a3_z3 = a3 * (1 - a3) # 1D per output neuron (3,)
        
        # C_z3 = C_b3 since z3_b3 = 1
        C_z3 = C_b3 = C_a3 * a3_z3 # 1D per output neuron (3,)
        
        # 2D weights incoming per output neuron (3,5)
        C_W3 = mat(C_z3, 1) * mat(a2, 0)
        
        # level 2 activations depend on all paths so we sum them.
        C_a2 = np.sum(W3.T * C_z3, axis=1) # (5,)
        
        C_z2 = C_b2 = C_a2 * f_deriv(a2) # (5,)
        
        C_W2 = mat(C_z2, 1) * mat(a1, 0)  # (5,5)
        
        C_a1 = np.sum(W2.T * C_z2, axis=1)  # (5,)
        
        C_z1 = C_b1 = C_a1 * f_deriv(a1)  # (5,)
        
        C_W1 = mat(C_z1, 1) * mat(a0, 0) # (5,13)
        
        # Add gradients to their respective sums
        if obs % test_period != 0:
            gW1 += C_W1
            gW2 += C_W2
            gW3 += C_W3
            gb1 += C_b1
            gb2 += C_b2
            gb3 += C_b3
            
      
    # Take average of gradients across observations
    m = X.shape[0]
    m = train_ctr
    gW1 /= m
    gW2 /= m
    gW3 /= m
    gb1 /= m
    gb2 /= m
    gb3 /= m
    
    # Make costs average
    train_cost /= train_ctr
    test_cost /= test_ctr
    
    # Update parameters
    W1 -= learning_rate * gW1
    W2 -= learning_rate * gW2
    W3 -= learning_rate * gW3
    b1 -= learning_rate * gb1
    b2 -= learning_rate * gb2
    b3 -= learning_rate * gb3
    
    if epoch % 20 == 0:
        p('{} Train. Cost - {:.05f}'.format(epoch, train_cost))
        p('{} Test.  Cost - {:.05f}'.format(epoch, test_cost))
        p('Train Accuracy : {}/{} = {:.2f}%'.format(train_correct, train_ctr, 100*train_correct/train_ctr))
        p('Test Accuracy  : {}/{} = {:.2f}%\n'.format(test_correct, test_ctr, 100*test_correct/test_ctr))


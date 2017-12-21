#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 16:21:54 2017

@author: pablorr10
"""

import numpy as np
import pandas as pd
import sklearn.preprocessing as preprop
import matplotlib.pyplot as plt
from statsmodels.tools.eval_measures import rmse

np.random.seed(2017)

class AddGate:
    '''
    Gate that performs addition between vectors --> They are gradient distributors
    '''
    def forward(self, q, b):
        # Compute the sumation q + b = r
        return q + b

    def backward(self, q, b, dJ):
        # Compute the chain rule 
        dq = dJ * np.ones_like(q) # Element-wise multiplication                                        # dJ/dq = (dJ/dr)·(dr/dq) = (dJ/dr)·1 --> Eq. B2
        ones = np.ones((1, dJ.shape[0]), dtype=np.float64)
        db = np.dot(ones, dJ)    # dJ/db = (dJ/dr)·(dr/db) = (dJ/dr)·1 --> Eq. B3
        return dq, db 
    

class MultiplyGate:
    '''
    Gate that performs multiplication between vectors --> They are gradient switchers
    '''
    def forward(self, X, W):
        # Compute the matrix multiplication q = X·W
        q = np.dot(X, W) 
        return q
    '''
    def backward(self, W, X, dJ):
        # Compute the chain rule (compute the bottom gradient, given the top gradient)
        dX = np.dot(dJ, np.transpose(W)) # dJ/dx = (dJ/dq)·(dq/dW) # dq/dX = transpose(W) --> Eq. B4
        dW = np.dot(np.transpose(X), dJ) # dJ/dW = (dJ/dq)·(dq/dX) # dq/dX = transpose(X) --> Eq. B5
        return dW, dX
    '''
    def backward(self, X, W, dJ):
        # Compute the chain rule (compute the bottom gradient, given the top gradient)
        dX = np.dot(dJ, W.T)  # dJ/dx = (dJ/dq)·(dq/dW) # dq/dX = transpose(W) --> Eq. B4
        dW = np.dot(X.T, dJ)  # dJ/dW = (dJ/dq)·(dq/dX) # dq/dX = transpose(X) --> Eq. B5
        return dX, dW

class Sigmoid:
    '''
    Neuron layer with the ability of applying a sigmoid function to its inputs
    '''
    def forward(self, X):
        # Apply the sigmpoid function     z = sigmoid(r)
        return 1.0 / (1.0 + np.exp(-X))             

    def backward(self, X, back_prop):
        # Compute the chain rule multiplying by its own derivative
        deriv = self.forward(X)                     # d(sigmoid(r)) 
        return (1.0 - deriv) * deriv * back_prop    # dJ/dr = (dJ/dz)·(dz/dr) = back_prop * dsigmoid --> Eq.B6


class Tanh:
    '''
    Neuron layer with the ability of applying a tanh function to its inputs
    '''
    # Apply the tanh function             z = tanh(r)
    def forward(self, X):
        return np.tanh(X)                           

    def backward(self, X, back_prop):
        # Compute the chain rule multiplying by its own derivative
        J = self.forward(X)                         # d(tanh(r))
        return (1.0 - np.square(J)) * back_prop     # dJ/dr = (dJ/dz)·(dz/dr) = back_prop * dtanh --> Eq.B6
    

class Cost:
    '''
    Class to calculate the cost based on the error commited when predicting
    '''
    def __init__(self, y_hat):
        # Receive the vector comming from the NN object
        self.y_hat = y_hat
    
    def loss(self, y_hat, y):   # Forward
        # Compute the sum of square errors to get the cost
        J = np.sum(np.square(y - y_hat))
        return J
    
    def diff(self, y):          # Backward
        # Compute the derivative of our cost to start the backProp process
        dJ = (-(y - self.y_hat)) 
        return dJ                       # return top_diff
        
    
'''
Neural Network
'''
class Net:
    '''
    Net that contains [Inputs, Gates, Layers, Activation Functions, Outputs]
    '''
    def __init__(self, inputLayerSize, hiddenLayerSize, outputLayerSize):
        # Random initialization of the weights. 
        # Our Matrix W has to convert from input dimension to layer dimension
        self.inputLayerSize  = inputLayerSize
        self.hiddenLayerSize = hiddenLayerSize
        self.outputLayerSize = outputLayerSize
        self.counter = 0
        
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        self.b1 = np.random.randn(self.hiddenLayerSize).reshape(1, -1)
        self.b2 = np.random.randn(self.outputLayerSize).reshape(1, -1)
        self.XW1  = 0
        self.z2   = 0
        self.a2   = 0
        self.a2W2 = 0
        self.z3   = 0
        self.cost = []
        '''
        # Create a matrix per layer for W and a vector per layer for b
        for i in range(len(self.layers_dim)-1):
            self.W.append(np.random.randn(self.layers_dim[i], self.layers_dim[i+1]) / np.sqrt(self.layers_dim[i]))
            self.b.append(np.random.randn(self.layers_dim[i+1]).reshape(1, self.layers_dim[i+1]))
        '''
            
        
    def feed_forward(self, X):
        '''
        Forward propagation of the input to get our prediction y_hat (or z)
        Definition of how or neural network computes
        '''        
        mulGate = MultiplyGate()
        addGate = AddGate()
        layer   = Sigmoid()
        
        # Now apply all the forward process for every neuron in the layer
        self.XW1   = mulGate.forward(X, self.W1)            # Eq. F1
        self.z2    = addGate.forward(self.XW1, self.b1)     # Eq. F2
        self.a2    = layer.forward(self.z2)                 # Eq. F3
        self.a2W2  = mulGate.forward(self.a2, self.W2)      # Eq. F4
        self.z3    = addGate.forward(self.a2W2, self.b2)    # Eq. F5       
        return self.z3
        
        
    def calculate_loss(self, z, y):
        '''
        Forward propagation to evaluate how well our model is doing
        '''
        costOutput = Cost(z)
        return costOutput.loss(z, y)                 # Eq. F6


    def train(self, X, y, epochs=20000, learning_rate=0.01, reg_lambda=0.01, print_loss=False):
        '''
        Batch gradient descent using the backpropagation algorithms
        '''
        mulGate    = MultiplyGate()
        addGate    = AddGate()
        layer      = Sigmoid()
    
        for epoch in range(epochs):           
            
            self.counter += 1           # Control Variable
            # Forward propagation
            z = self.feed_forward(X)  # Forward and update the value of all the variables
            costOutput = Cost(z)
            
            # Cost
            J = self.calculate_loss(z, y)
            self.cost.append(J)

            # Backward propagation
            top_diff   = costOutput.diff(y)                             # Eq. B1
            da2W2, db2 = addGate.backward(self.a2W2, self.b2, top_diff) # Eq. B2 + B3
            da2, dW2   = mulGate.backward(self.a2, self.W2, da2W2)      # Eq. B4 + B5
                # '''Layer'''
            dz2        = layer.backward(self.z2, da2)                   # Eq. B6  
                # '''Layer'''
            dXW1, db1  = addGate.backward(self.XW1, self.b1, dz2)       # Eq. B7 + B8
            dX1, dW1   = mulGate.backward(X, self.W1, dXW1)             # Eq. B9 + B10
            
            # Update weights
                
            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW1 += reg_lambda * self.W1
            dW2 += reg_lambda * self.W2
            
            # Gradient descent parameter update
            self.b1 += -learning_rate * db1
            self.W1 += -learning_rate * dW1
            self.b2 += -learning_rate * db2
            self.W2 += -learning_rate * dW2

            if print_loss and epoch % 100 == 0:
                print("Loss after iteration %i: %f" %(epoch, J))
                
            if epoch == epochs-1:

                plt.figure()
                plt.title('Evolution of Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.plot(self.cost, 'b-')
                plt.show()



'''
Problem for the NN
'''
# Create the inputs
dataset = np.random.randn(20, 2)*10
df_dataset = pd.DataFrame(dataset)
df_dataset.head()
 
X1 = (np.array(dataset[:,0])).reshape(-1, 1)
X2 = (np.array(dataset[:,1])).reshape(-1, 1)
Y = 0.5*X1 + np.square(X2*0.1) + 5*np.random.randn(1)

def scale(vector):
    # Function to scale a vector to range [0 1]
    maxs = np.max(vector)
    mins = np.min(vector)
    scaling = np.zeros(len(vector))
    
    for i in range(0, len(vector)):
        scaling[i] = (vector[i] - mins) / (maxs - mins)
    return scaling
    

def unScale(scal, unscal):
    # Unscale function to a vector giving the scaled and an unscaled reference vector
    maxs = np.max(unscal)
    mins = np.min(unscal)
    unscaling = np.zeros(len(scal))
    
    for i in range(0, len(scal)):    
        unscaling[i] = scal[i] * (maxs-mins) + mins
    return unscaling

'''
X1, X2 = np.meshgrid(X1, X2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X1, X2, Y, rstride=10, cstride=10)
plt.show()
'''

fig = plt.figure()
df_Y = pd.DataFrame(Y)
dataframe = pd.concat([df_dataset[0], df_dataset[1], df_Y], axis=1)
dataframe.plot()

'''
Problem for the NN
'''
scalerX1 = preprop.MinMaxScaler()
scalerX2 = preprop.MinMaxScaler()
scalerY  = preprop.MinMaxScaler()

x1 = scalerX1.fit_transform(X1)
x2 = scalerX2.fit_transform(X2)
y  = scalerY.fit_transform(Y)

x = np.concatenate((x1, x2), axis=1)

network = Net(inputLayerSize=2, hiddenLayerSize=3, outputLayerSize=1)
network.train(x, y, epochs=2000, learning_rate=0.01, reg_lambda=0.01, print_loss=True)

predictions = network.feed_forward(x)
predictions = scalerY.inverse_transform(predictions)
RMSE = rmse(predictions, Y)
print('RMSE = %.2f' % (RMSE[0]))

plt.figure()
plt.title('Actual vs Predicted')
plt.ylabel('Y')
plt.xlabel('X')
plt.plot(Y, 'b-', label='Actual')
plt.plot(predictions, 'r-', label='Predicted')
plt.legend()
plt.show()










           
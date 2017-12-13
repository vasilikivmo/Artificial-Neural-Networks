#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 16:21:54 2017

@author: pablorr10
"""

import numpy as np


class MultiplyGate:
    '''
    Gate that performs multiplication between vectors --> They are gradient switches
    '''
    def forward(self, X, W):
        # Compute the matrix multiplication q = X·W
        q = np.dot(X, W)
        return q

    def backward(self, W, X, dL):
        # Compute the bottom gradient, given the top gradient
        dW = np.dot(np.transpose(X), dL) # dL/dW = (dL/dq)·(dq/dX) # dq/dX = transpose(X) --> Eq. 14
        dX = np.dot(dL, np.transpose(W)) # dL/dx = (dL/dq)·(dq/dW) # dq/dX = transpose(W) --> Eq. 13
        return dW, dX


class AddGate:
    '''
    Gate that performs addition between vectors --> They are gradient distributors
    '''
    def forward(self, q, b):
        # Compute the sumation r = q + b = XW + b
        return q + b

    def backward(self, q, b, dL):
        dq = dL * np.ones_like(q)                   # dL/dq = (dL/dr)·(dr/dq) = (dL/dr)·1 --> Eq. 12
        db = np.dot(np.ones((1, dL.shape[0]), 
                             dtype=np.float64), dL) # dL/db = (dL/dr)·(dr/db) = (dL/dr)·1 --> Eq. 11
        return db, dq 
    '''Why different way of calculation?'''
    

class Sigmoid:
    '''
    Neuron layer with the ability of applying a sigmoid function to its inputs
    '''
    def forward(self, X):
        # Apply the sigmpoid function
        return 1.0 / (1.0 + np.exp(-X))             # d(sigmoid) --> Eq. 8

    def backward(self, X, top_diff):
        # Calculate the gradient backward that function
        deriv = self.forward(X)                     # d(sigmoid) 
        return (1.0 - deriv) * deriv * top_diff     # dL/dr = (dL/dz)·(dz/dr) = top_diff * dsigmoid --> Eq.10

class Tanh:
    '''
    Neuron layer with the ability of applying a tanh function to its inputs
    '''
    # Apply the sigmpoid function
    def forward(self, X):
        return np.tanh(X)                           # d(tanh) --> Eq. 8

    def backward(self, X, top_diff):
        # Calculate the gradient backward that function
        L = self.forward(X)                    # d(tanh)
        return (1.0 - np.square(L)) * top_diff # dL/dr = (dL/dz)·(dz/dr) = top_diff * dtanh --> Eq.10
    

class Cost:
    '''
    Class to calculate the cost based on the error commited when predicting
    '''
    def __init__(self, y_hat):
        # Receive the vector comming from the NN object
        self.y_hat = y_hat
    
    def loss(self, y):
        # Compute the sum of square errors to get the cost
        L = np.sum(np.square(y-self.y_hat))
        return L
    
    def diff(self, y):
        # Compute the derivative of our cost to start the backProp process
        dL = np.sum(-(y - self.y_hat))
        return dL                       # return top_diff
        
    
'''
Neural Network
'''
class Net:
    '''
    Net that contains [Inputs, Gates, Layers, Activation Functions, Outputs]
    '''
    def __init__(self, inputLayerSize, hiddenLayerSizes, outputLayerSize):
        # Random initialization of the weights. 
        # Our Matrix W has to convert from input dimension to layer dimension
        self.layers_dim = [inputLayerSize, hiddenLayerSizes, outputLayerSize]
        self.W = []
        self.b = []
        self.q = 0
        self.r = 0
        self.z = 0
        for i in range(len(self.layers_dim)-1):
            self.W.append(np.random.randn(self.layers_dim[i], self.layers_dim[i+1]) / np.sqrt(self.layers_dim[i]))
            self.b.append(np.random.randn(self.layers_dim[i+1]).reshape(1, self.layers_dim[i+1]))
        
    def feed_forward(self, X):
        '''
        Forward propagation of the input to get our prediction y_hat (or z)
        '''
        mulGate = MultiplyGate()
        addGate = AddGate()
        layer = Tanh()
        
        # Now apply all the forward process for every neuron in the layer
        for i in range(len(self.W)): # hiddenLayerSize?
            self.q = mulGate.forward(self.W[i], X)          # q = X·W  --> Eq. F1
            self.r = addGate.forward(self.q, self.b[i])     # r = q+b  --> Eq. F2
            self.z = layer.forward(self.r)                  # z = f(r) --> Eq. F3
        
        return self.z
        
    def calculate_loss(self, z, y):
        '''
        Forward propagation to evaluate how well our model is doing
        '''
        costOutput = Cost(z)
        return costOutput.loss(z, y)                    # L = cost(z,t) --> Eq. F4

    def train(self, X, y, epochs=20000, learning_rate=0.01, reg_lambda=0.01, print_loss=False):
        '''
        Batch gradient descent using the backpropagation algorithms
        '''
        mulGate = MultiplyGate()
        addGate = AddGate()
        layer = Tanh()
        costOutput = Cost()
    
        for epoch in range(epochs):           
            
            # Forward propagation
            z = self.feed_forward(X)
            
            # Cost
            L = self.calculate_cost(z, y)

            # Backward propagation
            top_diff = costOutput.diff(y)                       # dL/dz --> Eq. B1
            dr = layer.backward(L, top_diff)                    # dL/dr = (dL/dz)·(dz/dr) --> Eq. B2
            db, dq = addGate.backward(self.q, self.b, dr)       # dL/dq and dL/db --> Eq. B3 and Eq. B4
            dW = mulGate.backward(self.W, self.X, dq)           # dL/dW = (dL/dq)·(dq/dW) --> Eq. B5
            
            # Update weights
            for i in range(len(forward)-1, 0, -1):
                
                ''' Left in this part'''
                # Add regularization terms (b1 and b2 don't have regularization terms)
                dW += reg_lambda * self.W[i-1]
                # Gradient descent parameter update
                self.b[i-1] += -learning_rate * db
                self.W[i-1] += -learning_rate * dW
    
            if print_loss and epoch % 1000 == 0:
                print("Loss after iteration %i: %f" %(epoch, self.calculate_loss(X, y)))
                
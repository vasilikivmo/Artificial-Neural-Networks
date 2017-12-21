#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:29:41 2017

@author: pablorr10
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy import optimize
import matplotlib.pyplot as plt
import time

class Neural_Network(object):
    
    def __init__(self):
        # Define the layers we are going to have and the neurons in each layer
        self.inputLayerSize = x_train.shape[1]
        self.hiddenLayerSize = 3
        self.outputLayerSize = 1 

        # Weights initialization 
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, X):
        # Propagate inputs through neurons
        self.z2 = np.dot(X, self.W1)       # Eq. 1
        self.a2 = self.sigmoid(self.z2)    # Eq. 2
        self.z3 = np.dot(self.a2, self.W2) # Eq. 3
        yHat = self.sigmoid(self.z3)       # Eq. 4
        return yHat

    def sigmoid(self, z):   
        # Sigmoid activaction function
        return 1 / (1+np.exp(-z))

    def dSigmoid(self, z):
        # Derivative of sigmoid function
        return np.exp(-z)/((1+np.exp(-z))**2)

    def tanh(self, z):
        return np.tanh(z)
    
    def dtanh(self, z):
        return (1 - np.tanh(x)^2)
    
    def costFunction(self, X, y):
        # Compute the Cost Function using weights already stored in class
        self.yHat = self.forward(X)
        
        J = np.sum((y - self.yHat)**2) / X.shape[0] # Eq. 6
        return J     

    def dCostFunction(self, X, y):
        # Compute derivative with respect to W1 and W2 
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat), self.dSigmoid(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3) / X.shape[0] # Eq. 7

        delta2 = np.dot(delta3, self.W2.T) * self.dSigmoid(self.z2)
        dJdW1 = np.dot(X.T, delta2) / X.shape[0]    # Eq. 8

        return dJdW1, dJdW2

    def getWeights(self):
        # Get W1 and W2
        return self.W1, self.W2
    
    def setWeights(self, W1, W2):
        # Update the weights W1 and W2
        self.W1 = W1
        self.W2 = W2
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.dCostFunction(X, y)
        return dJdW1, dJdW2
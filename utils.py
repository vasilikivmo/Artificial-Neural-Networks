#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 18:14:43 2017

@author: pablorr10
"""

# Helper function to plot a decision boundary.
def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    
    
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
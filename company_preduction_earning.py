#!/usr/bin/env python
# coding: utf-8

# In[31]:


import os
from pprint import pprint
import pandas as pd
import numpy as np
#from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy import exp, array, random, dot


# In[32]:


import numpy as np

# X = (outcome, income), y = totalEarn
X = np.array(([1000, 500], [1000, 400], [900, 700]), dtype=float) # company income and outcome
y = np.array(([500], [600], [200]), dtype=float) # totalEarn 

# scale units
X = X/np.amax(X, axis=0) # maximum of X array
y = y 


# In[39]:


class Neural_Network(object):
  def __init__(self):
    #parameters
    self.inputSize = 2
    self.outputSize = 1
    self.hiddenSize = 50
    #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix from input to hidden layer
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer

  def forward(self, X):
    #forward propagation through our network
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
    self.z2 = self.sigmoid(self.z) # activation function
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
    o = self.sigmoid(self.z3) # final activation function
    return o

  def sigmoid(self, s):
    # activation function
    return 1/(1+np.exp(-s))

NN = Neural_Network()

#defining our output
o = NN.forward(X)*1000

print("Predicted Output: " + str(o))
print ("Actual Output: " + str(y))


# In[13]:


def sigmoidPrime(self, s):
  #derivative of sigmoid
  return s * (1 - s)


# In[14]:


def backward(self, X, y, o):
  # backward propagate through the network
  self.o_error = y - o # error in output
  self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

  self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
  self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

  self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
  self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights


# In[15]:


def train (self, X, y):
  o = self.forward(X)
  self.backward(X, y, o)


# In[16]:


for i in range(1000): # trains the NN 1,000 times
  print ("Input: " + str(X))
  print ("Actual Output: " + str(y))
  print ("Predicted Output: " + str(NN.forward(X)))
  print ("Loss: " + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
  NN.train(X, y)


# In[ ]:





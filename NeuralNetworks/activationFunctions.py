import numpy as np


"""
Series of activation functions. References to either papers or short summaries of papers that may or may not include sources
are added below. The derivatives of the activation functions are defined as well. They are labeled as the function+"Prime".
Summaries: http://cs231n.github.io/neural-networks-1/
Maxout Paper: https://arxiv.org/pdf/1302.4389.pdf
"""
def sigmoid(z):
  return 1.0 / (1.0 + np.exp(-z))

def sigmoidPrime(z):
  val = sigmoid(z)
  return val * (1-val)

def tanh(z): 
  return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z) )
  
def tanhPrime(z):
  return 1 - tanh(z)**2

def ReLU(z):
  return np.maximum(0,z)

def ReLUPrime(z):
  return 1 if z > 0 else 0

def LeakyReLU(z, slope): # this makes a small slope for the region < 0. Inconsistent results, but claimed to be better than ReLU
  return z if z > 0 else slope*z

def LeakyReLUPrime(z, slope):
  return 1 if z >0 else slope


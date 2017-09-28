import numpy as np

""" The d(cost)/d(activation) depends on the cost/loss function. So for the 'quadratic cost function', 
the derivative, is simply the difference of the final activation and answers. Returns a np.array of size
(nExamples x nClasses). """
def quadraticCost(output, answers):
  return np.sqrt((output - answers).sum(axis=1))

def quadraticCostPrime(output, answers):
    return (answers - output)


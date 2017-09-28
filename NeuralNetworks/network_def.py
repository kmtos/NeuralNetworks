#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np
import pandas as pd
import math
class Network(object):

  """The list ``sizes`` contains the number of neurons in the respective layers of the network. For example, if the 
  list  was [2, 3, 1] then it would be a three-layer network, with the first layer containing 2 neurons, the second 
  layer 3 neurons, and the third layer 1 neuron.  The biases and weights for the network are initialized randomly, 
  using a Gaussian distribution with mean 0, and variance 1.  Note that the first layer is assumed to be an input 
  layer, and by convention we won't set any biases for those neurons, since biases are only ever used in computing 
  the outputs from later layers. "self.biases" will contain as many 2-element tuples as maxoutLayers, where tup[0] is
  the maxout layer #, and the tup[1] will be the biases for the hidden and output layers via np.arrays with elements
  defined by self.sizes.  "self.weights" is structured the same way as "self.biases" except that instead of a 1D-vector
  value for each node, there is a JxK size matrix in each np.array, where j is the node in current_layer-1 and k is
  the node in current_layer."""
  def __init__(self, sizes, maxoutLayers, activFunc, activFuncPrime, costFunc, constFuncPrime, idColumn, classColumn, 
               learningRate, uniqueClassVals, additionalActivArgs):
    self.num_layers = len(sizes)
    self.sizes = sizes
    self.eta = learningRate
    self.additionalActivArgs = additionalActivArgs
    self.activationFuncPrime = activFuncPrime
    self.activationFunc = activFunc
    self.costFunc = costFunc
    self.constFuncPrime = constFuncPrime
    self.uniqueClasses = uniqueClassVals
    self.biases = []
    r = 4 *  math.sqrt(6/(2*30))
    for k in range(maxoutLayers):
      TEMP = [*np.random.randn(y, 1) for y in sizes[1:]]
      self.biases.append( (k,TEMP) )
    self.weights = []
    for k in range(maxoutLayers): 
      TEMP = [.001*np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
      self.weights.append( (k,TEMP) )
    self.idColumn = idColumn
    self.classColumn = classColumn

  """Train the neural network using mini-batch stochastic gradient descent.  The ``training_data`` is a dataframe
  representing the training inputs and the desired  outputs. The other non-optional parameters are self-explanatory.  
  If ``df_validation`` is provided then the network will be evaluated against the test data after each epoch, and 
  partial progress printed out.  This is useful for tracking progress, but slows things down substantially."""
  def SGD(self, df, epochs, miniBatchSize, df_validation=pd.DataFrame()):
    n = len(df)
    indecies = df[self.idColumn].tolist()
    random.shuffle(indecies)
    for epoch in range(epochs): 
      miniBatches = [ df[ df[self.idColumn].isin(indecies[k:k + miniBatchSize])] for k in range(0, n, miniBatchSize)]
      for df_miniBatch in miniBatches: 
        self.miniBatchBackProp_noMaxout(df_miniBatch)
      print ("Epoch", epoch, "complete")
      if not df_validation.empty: print ("\tEpoch #", epoch, ":", self.evaluate(df_validation), "/", len(df_validation) )

  """Update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch.
  The ``mini_batch`` is a list of tuples ``(x, y)``, and learning rate."""
  def miniBatchBackProp_noMaxout(self, df_miniBatch):
    change_b = [np.zeros(b.shape) for b in self.biases[0][1] ]
    change_w = [np.zeros(w.shape) for w in self.weights[0][1] ]
    """ The shapes of the matricies are as follows: 
    activation = (nInputs x nExamples) ||  np_answers = (nUniqueClasses x nExamples) || biases[i] = (nNodes in layer i x 1)
    weights[i] = (nNodes in layer i x nNodes in i-1) ||  delta = (nExamples X nNodes in outputLayer)
    delt = (nNodes in current layer x 1) 
    zs[i] = (nNodes in layer i x nExamples) || (act * delt) = (weights[i] for each layer)"""
    activation = np.transpose(df_miniBatch.drop([self.idColumn, self.classColumn], axis=1).values) 
    answers = df_miniBatch[[self.classColumn]].copy()
    for val in self.uniqueClasses: 
      answers[self.classColumn + "_" + str(val)] = answers[self.classColumn].map( lambda x: 1 if x == val else 0)
    np_answers = answers.drop([self.classColumn], axis=1).values
    zs = []
    activations = [activation]
    for b, w in zip(self.biases[0][1], self.weights[0][1]):
      z = np.dot(w,activation) + b
      zs.append(z)
      if self.additionalActivArgs == None: activation = self.activationFunc(z)
      else: activation = self.activationFunc(z, self.additionalActivArgs)
      activations.append(activation)
    if self.additionalActivArgs == None:
      delta = self.constFuncPrime(activations[-1].transpose(), np_answers) * np.transpose(self.activationFuncPrime(zs[-1]))
    else:
      delta = self.constFuncPrime(activations[-1].transpose(), np_answers) * np.transpose(
              self.activationFuncPrime(zs[-1], self.additionalActivArgs))
    for nEx in range(len(df_miniBatch.index)):
      act = np.array( [activations[-2].transpose()[nEx]] )
      delt = np.array( [delta[nEx]] ).transpose()
      change_w[-1] += (act * delt)
      change_b[-1] += delt
      for l in range(2,self.num_layers):
        zsCurr = np.array( [zs[-l].transpose()[nEx]]).transpose()
        if self.additionalActivArgs == None:
          delt = np.dot(self.weights[0][1][-l+1].transpose(), delt) * self.activationFuncPrime(zsCurr)
        else:
          delt = np.dot(self.weights[0][1][-l+1].transpose(), delt) * self.activationFuncPrime(zsCurr, self.additionalActivArgs)
        change_b[-l] += delt
        change_w[-l] += np.dot(delt, np.array( [activations[-l-1].transpose()[nEx]]) )

    new_weights = self.weights[0][1]
    new_biases = self.biases[0][1]
    new_weights = [w-(self.eta/len(df_miniBatch))*nw for w, nw in zip(new_weights, change_w)]
    new_biases = [b-(self.eta/len(df_miniBatch))*nb  for b, nb in zip(new_biases, change_b)]
    self.weights = [(0, new_weights)]
    self.biases = [(0, new_biases)]

  """Return the number of test inputs for which the neural network outputs the correct result. Note that the neural network's
   output is assumed to be the index of whichever neuron in the final layer has the highest activation."""
  def evaluate(self, df_validation):
    droppedColumns = df_validation[[self.idColumn, self.classColumn]].copy()
    columns = df_validation.columns
    answers = df_validation[self.classColumn].tolist()

    activation = np.transpose(df_validation.drop([self.idColumn, self.classColumn], axis=1).values)
    for b, w in zip(self.biases[0][1], self.weights[0][1]):
      z = np.dot(w,activation) + b
      if self.additionalActivArgs == None: activation = self.activationFunc(z)
      else: activation = self.activationFunc(z, self.additionalActivArgs)

    decisions = np.argmax(activation.transpose(), axis=1)
    print ("ACTIVATIONS:", activation.transpose() )
    print ("DECISION:\n", decisions)
    print ("answers:\n", answers)
    finalSum = sum(int(x == y) for x,y in zip(decisions, answers) )
    print("FINAL SCORE=", finalSum, "\tTOTAL=", len(df_validation.index))
    return finalSum

  """ For inputs x, compute the ouput activation ("activ") based upon the defined function (self.activationFunc) """
  def feedForward_noMaxout(self, inputs):
    for bias, weights in zip(self.biases[0][1], self.weights[0][1]):
      activ = self.activationFunc(np.dot(weights, a)+b, *self.additionalActivArgs)
    return activ

  def WriteWeights(self, fileName, separater):
     np.savetxt(fileName, self.weights, newline=separater)

  def WriteBiases(self, fileName, separater):
     np.savetxt(fileName, self.biases, newline=separater)

  def WriteNNOutput(self, fileName, separater):    
     activations = self.feedForward_noMaxout()
     np.savetxt(fileName, activations, newline=separater) 

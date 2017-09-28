import string

import numpy as np
import csv
import pandas as pd
import sys
from collections import defaultdict


sizes = [30,15,8,4,2]

biases = [np.random.randn(y, 1) for y in sizes[1:]]

weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

df = pd.read_csv('train_normed.csv', header=0)

answers = df[['Survived']].copy()


for val in answers['Survived'].unique():
     answers[val] = answers['Survived'].map( lambda x: 1 if x == val else 0)
    

np_answers = answers.drop(['Survived'], axis=1).values

def quadraticCostPrime(output, answers):
    return (answers - output)



def sigmoid(z):
  return 1.0 / (1.0 + np.exp(-z))

def sigmoidPrime(z):
  val = sigmoid(z)
  return val * (1-val)
 
activation = np.transpose(df.drop( ['Survived', 'PassengerId'], axis=1).values)                          
zs = []                                 
activations = [activation]                                                  
for b,w in zip(biases, weights):             
       print ("WSHAPE=", w.shape, "\naSHAPE=",activation.shape)
       z = np.dot(w,activation) + b                                                                                         
       print ("SHAPE OF Z=", z.shape)
       zs.append(z)                                                                                                         
       activation = 1.0 / (1.0 + np.exp(-z))                             
       activations.append(activation)
print ("DONE")         
                                                                
new_b = [np.zeros(b.shape) for b in biases ]
new_w = [np.zeros(w.shape) for w in weights ]


n = len(df)


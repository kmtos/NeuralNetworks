import string
import numpy as np
import csv
import pandas as pd
import sys
from collections import defaultdict
sys.path.insert(0, '/home/kyletos/Projects/NN/NeuralNetworks/')
#from subprocess import check_call
#from PIL import Image, ImageDraw, ImageFont
#from IPython.display import Image as PImage
import math
#import re

pd.set_option('display.max_columns', 100000000000)
pd.set_option('display.max_rows', 1000000000000)

#Setting the stdout file
orig_stdout = sys.stdout
stdOutFile = open('Answers/DecisionTree_stdOut.txt', 'w')
sys.stdout = stdOutFile

#Readinig in the csv file
df_ORI = pd.read_csv('train_normed.csv',header=0)

df_valid = df_ORI.sample(math.ceil(len(df_ORI.index) * (.1) ) ) #Selecting random portion of rows for double randomness
testIDs = df_valid['PassengerId'].tolist()
df = df_ORI[ ~df_ORI['PassengerId'].isin(testIDs) ]
print (len(df_valid.index))
print (len(df.index), "\n\n")
from network_def import *
from costFunctions import *
from activationFunctions import *
network1 = Network(sizes=[30,100,2], maxoutLayers=1, activFunc=sigmoid, activFuncPrime=sigmoidPrime, 
                   costFunc=quadraticCost, constFuncPrime=quadraticCostPrime, idColumn='PassengerId', classColumn='Survived', 
                   learningRate=.01, uniqueClassVals=[0.1], additionalActivArgs=None)

#print ("WEIGHTS:\n", network1.weights[0][1][-1])
#print ("\n\nBIASES:\n", network1.biases)
weight_ORI = network1.weights[0][1][-1]
network1.SGD(df=df, epochs=10000, miniBatchSize=math.ceil(len(df.index)/10.0), df_validation=df_valid)
weight_FIN = network1.weights[0][1][-1]

diff = weight_FIN - weight_ORI
#print (diff)
#print ("WEIGHTS:\n", network1.weights[0][1][-1])
#print ("\n\nBIASES:\n", network1.biases)



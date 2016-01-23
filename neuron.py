from __future__ import print_function

import sys
import glob
import os

import random

import numpy as np

class Neuron:
	def __init__(self, weights, activationFunction):
        self._weights = weights
		self._activationFunction = activationFunction
 	
    @property
	def weights(self):
		return self._weights

	
class NeuralLayer:
	def __init__(self, numNeurons, layerInput, neuronWeightsInLayer, activationFunction):
        self._numNeurons = numNeurons
		self._layerInput = layerInput
		self._layerOutput = None
		self._activationFunction = activationFunction
		self._neurons = [Neuron(self._layerInput, self._neuronWeightsInLayer[i], activationFunction) for i in range(numNeurons)]
	
    @property
	def numNeurons(self):
		return self._numNeurons

	@property
	def layerOutput(self):
		return self._layerOutput

class NeuralNetwork:
	def __init__(self, ipLayer, numLayers, numNeuronsPerLayer, weightMatrix=[], activationFunctions=[]):
		if len(numNeuronsPerLayer) != numLayers:
			print("Error: ")
        self._numLayers = numLayers
		self._layers = list()
		if weightMatrix==[]:
			self._weightMatrix = [[[random.uniform(0.0,1.0)]*numNeuronsPerLayer[i-1] for j in range(numNeuronsPerLayer[i])] for i in range(1, numLayers)] 
		else:
			if len(weightMatrix)!=numLayers:
				print("Error: ")
				sys.exit(0)
			else:
				for i in range(1, numLayers):
					if numNeuronsPerLayer[i]!=len(weightMatrix[i]):
						print("Error: ")
						sys.exit(0)
					else:
						for j in range(numNeuronsPerLayer[i]):
							if len(weightMatrix[i][j])!=numNeuronsPerLayer[i-1]:
								print("Error: ")
								sys.exit(0)
			self._weightMatrix = weightMatrix
		if activationFunctions==[]:
			self._activationFunctions = [Threshold() for i in range(numLayers-1)]
		else:
			if len(activationFunctions)!=numLayers-1:
				print("Error: ")
				sys.exit(0)
			else:
				self._activationFunctions = activationFunctions

		self._layers = [NeuralLayer(numNeuronsPerLayer[1], ipLayer, self._weightMatrix[0], self._activationFunctions[0]), NeuralLayer(numNeuronsPerLayer[i], [None]*numNeuronsPerLayer[i-1], self._weightMatrix[i], self._activationFunctions[i]) for i in range(2, numLayers)]

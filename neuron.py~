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
	def __init__(self, ipLayer, numLayers, numNeuronsPerLayer, weightMatrix=None, activationFunctions=None):
		self._ipLayer = ipLayer
		if len(numNeuronsPerLayer) != numLayers:
			print("Error: incompitable number of neurons")
			sys.exit(0)
		else:
			self._numNeuronsPerLayer = numNeuronsPerLayer
		self._numLayers = numLayers-1
		if weightMatrix==None:
			self._weightMatrix = [[[random.uniform(0.0,1.0)]*numNeuronsPerLayer[i-1] for j in range(numNeuronsPerLayer[i])] for i in range(1, numLayers)] 
		else:
			if len(weightMatrix)!=self._numLayers:
				print("Error: Insufficient weights assigned")
				sys.exit(0)
			else:
				for i in range(self._numLayers):
					if numNeuronsPerLayer[i+1]!=len(weightMatrix[i]):
						print("Error: Insufficient weights in Layer "+str(i+1))
						sys.exit(0)
					else:
						for j in range(numNeuronsPerLayer[i+1]):
							if len(weightMatrix[i][j])!=numNeuronsPerLayer[i]:
								print("Error: Insufficient weights provided for neuron "+str(j+1)+" in Layer "+str(i+1))
								sys.exit(0)
			self._weightMatrix = weightMatrix
		self._weightMatrix = np.matrix(weightMatrix)
		if activationFunctions==None:
			self._activationFunctions = [Threshold(1) for i in range(self._numLayers)]
		else:
			if len(activationFunctions)!=self._numLayers:
				print("Error: Insufficient Activation Functions provided")
				sys.exit(0)
			else:
				self._activationFunctions = activationFunctions

		self._layers = [NeuralLayer(numNeuronsPerLayer[1], ipLayer, self._weightMatrix[0], self._activationFunctions[0])] + [NeuralLayer(numNeuronsPerLayer[i], [None]*numNeuronsPerLayer[i-1], self._weightMatrix[i], self._activationFunctions[i]) for i in range(2, numLayers)]

	def forwardPropagation(self):
		a = self._ipLayer
		aMtrx = np.matrix(a)
		for i in range(self._numLayers):
			opMatrix = list(self._numNeuronsPerLayer[i])
			for j in range(self._numNeuronsPerLayer[i]):
				aggr = (aMtrx * self._weightMatrix)[0]
				opMatrix[j] = self._activationFunctions[i].compute(aggr)
			aMtrx = opMatrix
		return aMatrx

	def backPropagation(self):
		pass

	

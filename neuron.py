from __future__ import print_function

import sys
import glob
import os

import random

import numpy as np

class Neuron:
	def __init__(self, weights, activationFunction):
		self._weights = weights.transpose()
		self._activationFunction = activationFunction
	
	@property
	def weights(self):
		return self._weights

	def updateWeights(self, weights):
		self._weights = weights.transpose()

	def compute(self, ip):
		aggr = (ip * self._weights)[0]
		op = self._activationFunction.compute(aggr)
		return op

class NeuralLayer:
	def __init__(self, numNeurons, neuronWeightsInLayer, activationFunction):
		self._numNeurons = numNeurons
		self._activationFunction = activationFunction
		self._neuronWeightsInLayer = neuronWeightsInLayer
		self._neurons = [Neuron(self._neuronWeightsInLayer[i], self._activationFunction) for i in range(self._numNeurons)]

	@property
	def numNeurons(self):
		return self._numNeurons

	def compute(self, ip):
		opMatrix = [0.0]*self._numNeurons
		for j in range(self._numNeurons):
			opMatrix[j] = self._neurons[j].compute(ip)
		return opMatrix

class NeuralNetwork:
	def __init__(self, numLayers, numNeuronsPerLayer, weightMatrix=None, activationFunctions=None):
		if len(numNeuronsPerLayer) != numLayers:
			print("Error: incompitable number of neurons")
			sys.exit(0)
		else:
			self._numNeuronsPerLayer = numNeuronsPerLayer
		self._numLayers = numLayers-1
		if weightMatrix==None:
			self._weightMatrix = [[np.matrix([random.uniform(0.0,1.0)]*numNeuronsPerLayer[i-1]) for j in range(numNeuronsPerLayer[i])] for i in range(1, numLayers)] 
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
		self._weightMatrix = [[np.matrix(weightMatrix[i][j]) for j in range(numNeuronsPerLayer[i+1])] for i in range(self._numLayers)]
		self._weightMatrix = np.matrix(weightMatrix)
		if activationFunctions==None:
			self._activationFunctions = [Threshold(1) for i in range(self._numLayers)]
		else:
			if len(activationFunctions)!=self._numLayers:
				print("Error: Insufficient Activation Functions provided")
				sys.exit(0)
			else:
				self._activationFunctions = activationFunctions
		self._layers = [NeuralLayer(numNeuronsPerLayer[1], self._weightMatrix[0], self._activationFunctions[0])] + [NeuralLayer(numNeuronsPerLayer[i], self._weightMatrix[i], self._activationFunctions[i]) for i in range(2, numLayers)]

	def forwardPropagation(self):
		a = self._ipLayer
		aMatrix = np.matrix(a)
		for i in range(self._numLayers):
			aMatrix = np.matrix(self._layers[i].compute(aMatrix))
		return aMatrix

	def backPropagation(self):
		pass

	def updatedWeight(self, #iparams, exampleCnt, alpha):
		delW = [[np.matrix([0]*numNeuronsPerLayer[i-1]) for j in range(numNeuronsPerLayer[i])] for i in range(1, numLayers)]
		for i in range(exampleCnt):
			delW.item(i) = delW.item(i) + backPropagation(iparams)
		weightMatrix = [[np.matrix([weightMatrix.item(j)-alpha*(delW.item(j)/m)]*numNeuronsPerLayer[i-1]) for j in range(numNeuronsPerLayer[i])] for i in range(1, numLayers)]
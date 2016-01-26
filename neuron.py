from __future__ import print_function

import sys
import glob
import os

from sigmoid import *
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
		aggr = (ip * self._weights).item(0)
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
	def __init__(self, numLayers, numNeuronsPerLayer, weightMatrix=[], activationFunctions=[]):
		if len(numNeuronsPerLayer) != numLayers:
			print("Error: incompitable number of neurons")
			sys.exit(0)
		else:
			self._numNeuronsPerLayer = numNeuronsPerLayer
		self._numLayers = numLayers-1
		if weightMatrix==[]:
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
								print(weightMatrix)
								print("Error: Insufficient weights provided for neuron "+str(j+1)+" in Layer "+str(i+1))
								sys.exit(0)
			self._weightMatrix = weightMatrix
		self._weightMatrix = [[np.matrix(self._weightMatrix[i][j]) for j in range(numNeuronsPerLayer[i+1])] for i in range(self._numLayers)]
		if activationFunctions==[]:
			self._activationFunctions = [Threshold(1) for i in range(self._numLayers)]
		else:
			if len(activationFunctions)!=self._numLayers:
				print("Error: Insufficient Activation Functions provided")
				sys.exit(0)
			else:
				self._activationFunctions = activationFunctions
		self._layers = [NeuralLayer(numNeuronsPerLayer[i+1], self._weightMatrix[i], self._activationFunctions[i]) for i in range(self._numLayers)]

	def forwardPropagation(self, ip):
		aMatrix = np.matrix(ip)
		activationValues = []
		for i in range(self._numLayers):
			aMatrix = np.matrix(self._layers[i].compute(aMatrix))
			activationValues += [aMatrix]
		return activationValues

	def finalOutput(self,ip):
		activationValues = self.forwardPropagation(ip)
		return activationValues[self._numLayers-1]

	def dot(self, x, y):
		if x.shape[1]!=y.shape[1]:
			print("Error: Dimensions do not match for dot")
			sys.exit(0)
		lenX = x.shape[1]
		z = [None]*lenX
		for i in range(lenX):
			z[i] = x.item(i) * y.item(i)
		return np.matrix(z)

	def backPropagation(self, ip, expectedOutput):
		ip = np.matrix(ip)
		expectedOutput = np.matrix(expectedOutput)
		activationValues = self.forwardPropagation(ip)
		delta = [None]*self._numLayers
		activationFunctionDerivative = [None]*self._numLayers
		for i in range(self._numLayers):
			activationFunctionDerivative[i] = np.matrix([(activationValues[i].item(j)*(1-activationValues[i].item(j))) for j in range(self._numNeuronsPerLayer[i+1])])
		
		# Set delta for output layer
		delta[self._numLayers-1] = self.dot((activationValues[self._numLayers-1] - expectedOutput), activationFunctionDerivative[self._numLayers-1])
		
		delW = [None]*self._numLayers
		# set delta for all layers
		for l in range(self._numLayers-2, -1, -1):
			aggr = [0.0]*self._numNeuronsPerLayer[l+1]
			for i in range(self._numNeuronsPerLayer[l+1]):
				for j in range(self._numNeuronsPerLayer[l+2]):
					aggr[i] += (self._weightMatrix[l][i] * delta[l+1].item(j)).item(0)
			aggr = np.matrix(aggr)
			delta[l] = self.dot(aggr, activationFunctionDerivative[l])
		
		for l in range(self._numLayers):
			delW[l] = [None]*self._numNeuronsPerLayer[l]
			if l==0:
				y = ip
			else:
				y = activationValues[l-1]
			delW[l] = (delta[l].transpose() * y)
		
		return delW

	def backPropagationUpdateWeight(self, batchInputs, expectedBatchOutputs, alpha):
		if len(batchInputs)!=len(expectedBatchOutputs):
			print("Error: Invalid input provided. Size of batchInputs and expectedBatchOutputs differ.")
			sys.exit(0)
		ipCount = len(batchInputs)
		for i in range(ipCount):
			ip = np.matrix(batchInputs[i])
			expectedOutput = np.matrix(expectedBatchOutputs[i])
			delW = self.backPropagation(ip, expectedOutput)
			for j in range(self._numLayers):
				for k in range(self._numNeuronsPerLayer[j+1]):
					self._weightMatrix[j][k] = self._weightMatrix[j][k] + delW[j][k]

		print("Updated WeightMatrix:")
		print(self._weightMatrix)













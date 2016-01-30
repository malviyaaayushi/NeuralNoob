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
	
	def weights(self):
		return self._weights.transpose()

	def updateWeights(self, weights):
		self._weights = weights.transpose()

	def compute(self, ip):
		aggr = (ip * self._weights).item(0)
		op = self._activationFunction.compute(aggr)
		#useful line for debugging
		#print("ip: "+str(ip)+" weights: "+str(self._weights.transpose())+" op: "+str(op))
		return op

class NeuralLayer:
	def __init__(self, numNeurons, neuronWeightsInLayer, activationFunctionsInLayer):
		self._numNeurons = numNeurons
		self._activationFunctionsInLayer = activationFunctionsInLayer
		self._neurons = [Neuron(neuronWeightsInLayer[i], self._activationFunctionsInLayer[i]) for i in range(self._numNeurons)]

	def numNeurons(self):
		return self._numNeurons

	def neurons(self):
		return self._neurons

	def compute(self, ip):
		opMatrix = [0.0]*self._numNeurons
		for j in range(self._numNeurons):
			opMatrix[j] = self._neurons[j].compute(ip)
		return opMatrix
	
	def updateWeights(self, newNeuronWeightsInLayer):
		for i in range(self._numNeurons):
			self._neurons[i].updateWeights(newNeuronWeightsInLayer[i])

class NeuralNetwork:
	def __init__(self, numLayers, numNeuronsPerLayer, weightMatrix=[], activationFunctions=[]):
		if len(numNeuronsPerLayer) != numLayers:
			print("Error: incompitable number of neurons")
			sys.exit(0)
		else:
			self._numNeuronsPerLayer = numNeuronsPerLayer
		self._numLayers = numLayers-1
		if weightMatrix==[]:
			weightMatrix = [[[random.uniform(0.0,1.0) for k in range(self._numNeuronsPerLayer[i])] for j in range(self._numNeuronsPerLayer[i+1])] for i in range(self._numLayers)] 
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
			weightMatrix = weightMatrix
		weightMatrix = [[np.matrix(weightMatrix[i][j]) for j in range(self._numNeuronsPerLayer[i+1])] for i in range(self._numLayers)]
		if activationFunctions==[]:
			self._activationFunctions = [[Threshold(1) for j in range(self._numNeuronsPerLayer[i+1])] for i in range(self._numLayers)]
		else:
			if len(activationFunctions)!=self._numLayers:
				print("Error: Insufficient Activation Functions provided")
				sys.exit(0)
			else:
				self._activationFunctions = activationFunctions
		self._layers = [NeuralLayer(self._numNeuronsPerLayer[i+1], weightMatrix[i], self._activationFunctions[i]) for i in range(self._numLayers)]

	def forwardPropagation(self, ip):
		aMatrix = ip
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

	def backPropagationUtil(self, expectedOutput, activationValues):
		delta = [None]*self._numLayers
		activationFunctionDerivative = [None]*self._numLayers
		for i in range(self._numLayers):
			activationFunctionDerivative[i] = np.matrix([(activationValues[i].item(j)*(1-activationValues[i].item(j))) for j in range(self._numNeuronsPerLayer[i+1])])
		
		# Set delta for output layer
		delta[self._numLayers-1] = self.dot((activationValues[self._numLayers-1] - expectedOutput), activationFunctionDerivative[self._numLayers-1])
		
		# set delta for all layers
		for l in range(self._numLayers-2, -1, -1):
			aggr = [0.0]*self._numNeuronsPerLayer[l+1]
			for i in range(self._numNeuronsPerLayer[l+1]):
				for j in range(self._numNeuronsPerLayer[l+2]):
					aggr[i] += (((self._layers[l].neurons())[i]).weights() * delta[l+1].item(j)).item(0)
			aggr = np.matrix(aggr)
			delta[l] = self.dot(aggr, activationFunctionDerivative[l])
		
		return delta

	def rmsError(self, expectedOutput, activationValues):
		e = 0.0
		iRange = self._numNeuronsPerLayer[self._numLayers]
		for i in range(iRange):
			e += pow(expectedOutput.item(i) - activationValues[self._numLayers-1].item(i), 2)
		return pow(e/iRange, 0.5)

	def updateWeights(self, newWeights):
		for i in range(self._numLayers):
			self._layers[i].updateWeights(newWeights[i])			

	def backPropagationUpdateWeight(self, ip, delta, activationValues):
		delW = [None]*self._numLayers
		for l in range(self._numLayers):
			delW[l] = [None]*self._numNeuronsPerLayer[l]
			if l==0:
				y = ip
			else:
				y = activationValues[l-1]
			delW[l] = (delta[l].transpose() * y)
		
		newWeightMatrix = [[None for j in range(self._numNeuronsPerLayer[i+1])] for i in range(self._numLayers)]

		for j in range(self._numLayers):
			for k in range(self._numNeuronsPerLayer[j+1]):
				newWeightMatrix[j][k] = self._layers[j].neurons()[k].weights() - delW[j][k]

		self.updateWeights(newWeightMatrix)
	
	
	def backPropagation(self, batchInputs, expectedBatchOutputs, alpha, e, verbose):
		if len(batchInputs)!=len(expectedBatchOutputs):
			print("Error: Invalid input provided. Size of batchInputs and expectedBatchOutputs differ.")
			sys.exit(0)
		ipCount = len(batchInputs)
		run = True
		while run:
			run = False
			for i in range(ipCount):
				ip = np.matrix(batchInputs[i])
				expectedOutput = np.matrix(expectedBatchOutputs[i])
				activationValues = self.forwardPropagation(ip)
				delta = self.backPropagationUtil(expectedOutput, activationValues)
				
				self.backPropagationUpdateWeight(ip, delta, activationValues)
				
				er = self.rmsError(expectedOutput, activationValues)
				if er>e:
					run = True

				if verbose:
					print("Current error: "+str(er))
					print("Current output: "+str(activationValues[self._numLayers-1]))
					print("Expected output: "+str(expectedOutput))
					print("\n")
		
		if verbose:
			print("Final Weights:")
			for j in range(self._numLayers):
				print("Layer "+str(j+1)+": ", end="")
				for k in range(self._numNeuronsPerLayer[j+1]):
					print(str(self._layers[j].neurons()[k].weights()), end="")
				print("")			

	def perceptronLearningUpdateWeight(self, ip, delta, activationValues, alpha):
		newWeightMatrix = [[None for j in range(self._numNeuronsPerLayer[i+1])] for i in range(self._numLayers)]
		for j in range(self._numLayers):
			for k in range(self._numNeuronsPerLayer[j+1]):
				newWeightMatrix[j+1][k] =np.add(self._layers[j+1].neurons()[k].weights(), alpha*delta*activationValues[j].item(k))
		self.updateWeights(newWeightMatrix)
	
	def perceptronLearning(self, batchInputs, expectedBatchOutputs, alpha, e, verbose):
		if len(batchInputs)!=len(expectedBatchOutputs):
			print("Error: Invalid input provided. Size of batchInputs and expectedBatchOutputs differ.")
			sys.exit(0)
		ipCount = len(batchInputs)
		run = True
		while run:
			run = False
			for i in range(ipCount):
				ip = np.matrix(batchInputs[i])
				expectedOutput = np.matrix(expectedBatchOutputs[i])
				activationValues = self.forwardPropagation(ip)
				delta = expectedOutput - activationValues[self._numLayers-1]
				
				self.perceptronLearningUpdateWeight(ip, delta, activationValues, alpha)
				
				er = self.rmsError(expectedOutput, activationValues)
				if er>e:
					run = True
				
				if verbose:
					print("Current error: "+str(er))
					print("Current output: "+str(activationValues[self._numLayers-1]))
					print("Expected output: "+str(expectedOutput))
					print("\n")
		
		if verbose:
			print("Final Weights:")
			for j in range(self._numLayers):
				print("Layer "+str(j+1)+": ", end="")
				for k in range(self._numNeuronsPerLayer[j+1]):
					print(str(self._layers[j].neurons()[k].weights()), end="")
				print("")

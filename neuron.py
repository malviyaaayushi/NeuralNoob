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
from neuron import *
from threshold import *
import numpy as np
class McCullochPitts:
	def __init__(self, ip, weights, t):
		self._ip = ip
		self._weights = np.matrix(weights)
		self._activationFunction = Threshold(t)
		self._neuralNetwork = NeuralNetwork(2, [self._ip ,1], self._weights, [self._activationFunction])

	def forwardPropagation(self,ip):
		return self._neuralNetwork.finalOutput(ip).item(0)

	def backPropagation(self):
		pass
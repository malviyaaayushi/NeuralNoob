from neuron import *
from threshold import *
import numpy as np
class McCullochPitts:
	def __init__(self, numIpFeatures, weights, t):
		self._weights = weights
		self._activationFunction = Threshold(t)
		self._neuralNetwork = NeuralNetwork(2, [numIpFeatures, 1], [[self._weights]], [self._activationFunction])

	def forwardPropagation(self,ip):
		return self._neuralNetwork.finalOutput(ip).item(0)

	def backPropagation(self):
		pass

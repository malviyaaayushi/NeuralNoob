from neuron import *
from threshold import *

class McCullochPitts:
	def __init__(self, ip, weights, t):
		self._ip = ip
		self._weights = weights
		self._activationFunction = Threshold(t)
		self._neuralNetwork = NeuralNetwork(self._ip, 2, [len(self._ip) ,1], [[self._weights]], [self._activationFunction])

	def forwardPropagation(self):
		return self._neuralNetwork.forwardPropagation()[0]
		self._neuralNetwork = NeuralNetwork(self._ip, 2, [len(self._ip) ,1], [[self._weights]], self._activationFunction)

	def forwardPropagation(self):
		return self._neuralNetwork.forwardPropagation()[0]

from neuron import *
from threshold import *

class McCullochPitts:
	def __init__(self, ip, weights, t):
		self._ip = ip
		self._weights = weights
		self._threshold = t
		self._neuralNetwork = NeuralNetwork(ip, 2, [1], [[weights]], Threshold(t))

	def forwardPropagation(self):
		

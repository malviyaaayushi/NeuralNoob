import math

class Sigmoid:
	def __init__(self):
		pass

	def derivative(self, x):
		return self.compute(x) * (1 - self.compute(x))
	
	def compute(self, x):
		return 1 / (1 + math.exp(-x))

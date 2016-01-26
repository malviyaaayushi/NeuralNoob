import math

class Sigmoid:
	def __init__(self):
		pass

	def derivative(self, x):
		fVal = self.compute(x)
		return fVal * (1 - fVal)
	
	def compute(self, x):
		return 1 / (1 + math.exp(-x))

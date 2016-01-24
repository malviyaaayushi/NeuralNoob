
class Threshold:
	def __init__(t):
		self._t = t
	
	def compute(self, aggr):	
		if aggr >= self._t:
			return 1
		return 0

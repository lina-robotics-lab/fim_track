import numpy as np

class Region:
	"""A Region must define a point_project method, which takes in a point and return its projected point."""
	def __init__(self):
		pass
	def point_project(self,pt):
		print('point_project is not defined')
		return None
		
class Rect2D(Region):
	"""A 2-D Rectangle"""
	def __init__(self, xlims=(0,0), ylims=(0,0)):
		super(Rect2D,self).__init__()
		self.xmin = np.min(xlims)
		self.xmax = np.max(xlims)
		self.ymin = np.min(ylims)
		self.ymax = np.max(ylims)

	def project_point(self,pt):
		def constrain(input, low, high):
			if input < low:
				input = low
			elif input > high:
				input = high
			else:
				input = input
			return input

		pt = np.array(pt).flatten()
		assert(len(pt)==2)

		return np.array([constrain(pt[0],self.xmin,self.xmax),\
						 constrain(pt[1],self.ymin,self.ymax)])
		
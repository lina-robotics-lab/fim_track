#!/usr/bin/env python3
import numpy as np
class virtual_sensor(object):
	"""
	Remember to use virtual_sensor.py (not just virtual_sensor) to refer to this package.

	The virtual sensor used in simulations. 
	"""
	def __init__(self,C1,C0,b,k,noise_std):

	
		self.C1=C1
		self.C0=C0
		self.b=b
		self.k=k
		self.noise_std= noise_std

	def measurement(self,source_locs,sensor_locs):
		"""
		The measurement model: y = k(r-C_1)^b+C_0
		
		Output: y satisfying len(y)=len(sensor_locs)
		"""

		# Use the outter product technique to get the pairwise displacement between sources and sensors
		q2p=source_locs[:,np.newaxis]-sensor_locs

		# calculate the pairwise distance
		d=np.linalg.norm(q2p,axis=-1)


		y = (self.k*(d-self.C1)**self.b)+self.C0

		# Sum over the influence of all sources to get the total measurement for each sensor.
		y=np.sum(y,axis = 0)


		# Add noise
		y += np.random.randn(*y.shape)*self.noise_std

		# Avoid pathological values in location estimation.
		y[y<=0]=1e-7

		return y

import numpy as np

from RemotePCCodebase import stop_twist

from geometry_msgs.msg import Twist


"""
	Simple Proportional Turn-and-Go controller
"""


# The default turtlebot speed is SUUUUPER slow.
# BURGER_MAX_LIN_VEL = 0.22
# BURGER_MAX_ANG_VEL = 2.84

BURGER_MAX_LIN_VEL = 0.22*10
BURGER_MAX_ANG_VEL = 2.84*10
turtlebot3_model='burger'

def constrain(input, low, high):
		if input < low:
		  input = low
		elif input > high:
		  input = high
		else:
		  input = input
		return input

class TurnAndGo:

	def __init__(self,linear_vel_gain=1.5,angular_vel_gain=6,reached_tolerance=0.1):
		self.loc=None
		self.yaw=None
		self.linear_vel_gain=linear_vel_gain
		self.angular_vel_gain=angular_vel_gain
		self.reached_tolerance=reached_tolerance
	
	def get_twist(self,target_loc,curr_loc,curr_yaw):
		self.loc=np.array(curr_loc).ravel()
		self.yaw=curr_yaw

		if np.linalg.norm(target_loc-self.loc)<=self.reached_tolerance:
			return stop_twist()
		"""
		This is a linear distance-based Proportional controller. v=Kv*||p-p_target||, w=Kw*|theta-target_yaw_to_me|
		"""
		vel_msg=Twist()
		# Linear velocity in the x-axis.
		
		vel_msg.linear.x = self._linear_vel(target_loc)
		vel_msg.linear.x = self._legal_linear_vel(vel_msg.linear.x)
		vel_msg.linear.y = 0
		vel_msg.linear.z = 0

		# Angular velocity in the z-axis.
		vel_msg.angular.x = 0
		vel_msg.angular.y = 0
		vel_msg.angular.z = self._angular_vel(target_loc)
		vel_msg.angular.z=self._legal_angular_vel(vel_msg.angular.z)
		
		return vel_msg
	
	def _legal_linear_vel(self, v):
		if turtlebot3_model == "burger":
		  v = constrain(v, -BURGER_MAX_LIN_VEL, BURGER_MAX_LIN_VEL)
		elif turtlebot3_model == "waffle" or turtlebot3_model == "waffle_pi":
		  v = constrain(v, -WAFFLE_MAX_LIN_VEL, WAFFLE_MAX_LIN_VEL)
		else:
		  v = constrain(v, -BURGER_MAX_LIN_VEL, BURGER_MAX_LIN_VEL)
		return v
	def _legal_angular_vel(self, w):
		if turtlebot3_model == "burger":
		  w = constrain(w, -BURGER_MAX_ANG_VEL, BURGER_MAX_ANG_VEL)
		elif turtlebot3_model == "waffle" or turtlebot3_model == "waffle_pi":
		  w = constrain(w, -WAFFLE_MAX_ANG_VEL, WAFFLE_MAX_ANG_VEL)
		else:
		  w = constrain(w, -BURGER_MAX_ANG_VEL, BURGER_MAX_ANG_VEL)
		return w
	
	def _linear_vel(self, target_loc):
		"""See video: https://www.youtube.com/watch?v=Qh15Nol5htM."""
		target_loc=np.array(target_loc).reshape(self.loc.shape)
		return self.linear_vel_gain * np.linalg.norm(target_loc-self.loc)

	def _steering_angle(self, target_loc):
		"""See video: https://www.youtube.com/watch?v=Qh15Nol5htM."""
		target_loc=np.array(target_loc).reshape(self.loc.shape)
		displacement=target_loc-self.loc
		return np.arctan2(displacement[1],displacement[0])

	def _angular_vel(self, target_loc):
		"""See video: https://www.youtube.com/watch?v=Qh15Nol5htM."""
		return self.angular_vel_gain * (self._steering_angle(target_loc) - self.yaw)
	
	
	
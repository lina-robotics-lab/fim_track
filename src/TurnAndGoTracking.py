import numpy as np

from RemotePCCodebase import stop_twist,angle_substract

from geometry_msgs.msg import Twist


"""
	Simple Proportional Turn-and-Go controller
"""


# The default turtlebot speed is SUUUUPER slow.
# BURGER_MAX_LIN_VEL = 0.22
# BURGER_MAX_ANG_VEL = 2.84

BURGER_MAX_LIN_VEL = 0.22 * 0.5
BURGER_MAX_ANG_VEL = 2.84 

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

	def __init__(self,angular_vel_gain=6,reached_tolerance=0.1):
		self.loc=None
		self.yaw=None

		# Ensure the gain is positive, and does not drive the omega above BURGER_MAX_ANG_VEL.
		self.angular_vel_gain=constrain(angular_vel_gain,0,BURGER_MAX_ANG_VEL/np.pi)
		
		self.reached_tolerance=reached_tolerance
	
	def get_twist(self,target_loc,curr_loc,curr_yaw):
		self.loc=np.array(curr_loc).ravel()
		self.yaw=curr_yaw
		target_yaw=self._steering_angle(target_loc)
		dist = np.linalg.norm(target_loc-self.loc)
		if dist<=self.reached_tolerance and np.abs(curr_yaw-target_yaw)<self.reached_tolerance*np.pi:
			return stop_twist()
		"""
		This is a linear distance-based Proportional controller. v=Kv*||p-p_target||, w=Kw*|theta-target_yaw_to_me|
		"""
		vel_msg=Twist()
		# Linear velocity in the x-axis.
		omega=self._angular_vel(target_yaw)
		
		vel_msg.linear.x = self._linear_vel(omega,dist)
		vel_msg.linear.x = self._legal_linear_vel(vel_msg.linear.x)
		vel_msg.linear.y = 0
		vel_msg.linear.z = 0

		# Angular velocity in the z-axis.
		vel_msg.angular.x = 0
		vel_msg.angular.y = 0
		vel_msg.angular.z = omega
		vel_msg.angular.z=self._legal_angular_vel(vel_msg.angular.z)

		# print(vel_msg)
		
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
	
	def _linear_vel(self,angular_vel,dist,K=0.5,v0=0.5):
		"""
			This ensures when angular velocity is large, the linear velocity
			will be small. When angular velocity is small, the robot can go
			linearly in speed that is dampened by its closeness to the target.
		"""
		return BURGER_MAX_LIN_VEL*(1-angular_vel/BURGER_MAX_ANG_VEL)*np.min([v0+np.abs(dist/K),1])

	def _steering_angle(self, target_loc):
		"""See video: https://www.youtube.com/watch?v=Qh15Nol5htM."""
		target_loc=np.array(target_loc).reshape(self.loc.shape)
		displacement=target_loc-self.loc
		return np.arctan2(displacement[1],displacement[0])

	def _angular_vel(self, target_yaw):
		"""See video: https://www.youtube.com/watch?v=Qh15Nol5htM."""
		# New version, taking into consideration the most efficient direction of turning.
		return self.angular_vel_gain * angle_substract(target_yaw , self.yaw)
		
		# Old version
		# return self.angular_vel_gain * (target_yaw - self.yaw)

	
	
	
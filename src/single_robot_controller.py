#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped,Pose, Twist
from std_msgs.msg import Float32MultiArray
import numpy as np
from functools import partial

from RemotePCCodebase import prompt_pose_type_string,toxy
from robot_listener import robot_listener
from collections import deque

BURGER_MAX_LIN_VEL = 0.22
BURGER_MAX_ANG_VEL = 2.84

class single_robot_controller(object):
	"""
	 single_robot_controller

	 Interactions: receive waypoints from multi_robot_controller. Send out cmd_vel commands to mobile sensors.
	 Also receive pose from the mobile sensors.

	 Controller Kernel Algorithm: LQR for waypoint tracking.
	"""
	def __init__(self, robot_name,pose_type_string,awake_freq=10):
		self.robot_name=robot_name
		self.awake_freq=awake_freq
		
		rospy.init_node('single_robot_controller_{}'.format(self.robot_name))
		self.listener=robot_listener(robot_name,pose_type_string)
		self.waypoint_sub=rospy.Subscriber('/{}/waypoints'.format(self.robot_name),Float32MultiArray,self.waypoint_callback_)
		
		self.all_waypoints=None
		self.remaining_waypoints=None

		self.kernal_algorithm='LQR'
		self.lqr_u=deque() # Should be a deque of twists


	def waypoint_callback_(self,data):
		self.all_waypoints=np.array(data.data).reshape(-1,2)
		self.remaining_waypoints=deque([self.all_waypoints[i,:] for i in range(len(self.all_waypoints))])
	
	def get_next_u(self):
		if self.kernal_algorithm=='LQR':
			if len(self.lqr_u)>0:
				return self.lqr_u[0]

	def start(self):
		rate=rospy.Rate(self.awake_freq)

		while not rospy.is_shutdown():
			print('single robot:',self.robot_name)
			if not(self.listener.robot_pose==None):					
				self.listener.robot_loc_stack.append(toxy(self.listener.robot_pose))
				# print(self.listener.robot_loc_stack[-1])	
			
			if not self.all_waypoints is None:
				# print(self.all_waypoints.shape)
				# print(self.remaining_waypoints[0],self.remaining_waypoints[-1])
				pass
			rate.sleep()

if __name__ == '__main__':
	pose_type_string=prompt_pose_type_string()
	
	robot_no=input('The index for this robot is:')
	
	robot_name='mobile_sensor_{}'.format(robot_no)
	
	controller=single_robot_controller(robot_name,pose_type_string)	
	controller.start()
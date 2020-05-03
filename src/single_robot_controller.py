#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped,Pose, Twist
from std_msgs.msg import Float32MultiArray
import numpy as np
from functools import partial
import sys

from RemotePCCodebase import prompt_pose_type_string,toxy,toyaw,stop_twist
from robot_listener import robot_listener
from collections import deque
from PathTrackingAlgs import TurnAndGo

from geometry_msgs.msg import Twist

class single_robot_controller(object):
	"""
	 single_robot_controller

	 Interactions: receive waypoints from multi_robot_controller. Send out cmd_vel commands to mobile sensors.
	 Also receive pose from the mobile sensors.

	 Controller Kernel Algorithm: LQR for waypoint tracking.
	"""
	def __init__(self, robot_name,pose_type_string,awake_freq=10,kernal_algorithm='LQR'):
		# Parameters
		self.robot_name=robot_name
		self.awake_freq=awake_freq
		self.kernal_algorithm=kernal_algorithm


		# Data containers
		self.all_waypoints=None
		self.remaining_waypoints=None
		self.lqr_u=deque() # Should be a deque of twists

		# ROS Setup
		rospy.init_node('single_robot_controller_{}'.format(self.robot_name))
		self.listener=robot_listener(robot_name,pose_type_string)
		self.waypoint_sub=rospy.Subscriber('/{}/waypoints'.format(self.robot_name),Float32MultiArray,self.waypoint_callback_)
		self.vel_pub=rospy.Publisher('/{}/cmd_vel'.format(self.robot_name),Twist,queue_size=10)
		
		

	def waypoint_callback_(self,data):
		self.all_waypoints=np.array(data.data).reshape(-1,2)
		self.remaining_waypoints=deque([self.all_waypoints[i,:] for i in range(len(self.all_waypoints))])
	
	def get_next_vel(self):
		vel_msg=stop_twist()

		if self.kernal_algorithm=='LQR':
			if len(self.lqr_u)>0:
				vel_msg=self.lqr_u.popleft()
		elif self.kernal_algorithm=='TurnAndGo':
			if len(self.remaining_waypoints)>1:
				loc=self.listener.robot_loc_stack[-1]
				yaw=self.listener.robot_yaw_stack[-1]
				target_loc=self.remaining_waypoints[0]
				vel_msg=TurnAndGo(linear_vel_gain=1.5,angular_vel_gain=6).get_twist(target_loc,loc,yaw)
				if vel_msg==stop_twist():
					self.remaining_waypoints.popleft()
		return vel_msg
	
	def update_remaining_waypoints(self):
		if not (self.remaining_waypoints is None):
			if len(self.remaining_waypoints)>0:
				
				print(self.remaining_waypoints.popleft())
				print(len(self.remaining_waypoints))

	def start(self):
		rate=rospy.Rate(self.awake_freq)

		try:
			while not rospy.is_shutdown():
				print('single robot:',self.robot_name)
				if not(self.listener.robot_pose==None):					
					self.listener.robot_loc_stack.append(toxy(self.listener.robot_pose))
					self.listener.robot_yaw_stack.append(toyaw(self.listener.robot_pose))
					print(self.listener.robot_yaw_stack[-1])	
				
				if not self.all_waypoints is None:
					vel_msg=self.get_next_vel()
					self.vel_pub.publish(vel_msg)
								
				rate.sleep()
		except:
			pass
		finally:
			self.vel_pub.publish(stop_twist())
	
		

if __name__ == '__main__':
	arguments = len(sys.argv) - 1
	

	if arguments<=0:
		pose_type_string=prompt_pose_type_string()
		robot_no=input('The index for this robot is:')
	else:
		if arguments>=1:
			pose_type_string=sys.argv[1]
		if arguments>=2:
			robot_no=int(sys.argv[2])

	
	robot_name='mobile_sensor_{}'.format(robot_no)
	
	controller=single_robot_controller(robot_name,pose_type_string,kernal_algorithm='TurnAndGo')	
	controller.start()
#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped,Pose, Twist
from std_msgs.msg import Float32MultiArray
import numpy as np

from RemotePCCodebase import prompt_pose_type_string,toxy
from robot_listener import robot_listener

class multi_robot_controller(object):
	"""

		multi_robot_controller

		Posession: a list of single_robot_controllers. 
			Each single_robot_controller has a sequence of waypoints to track. 
			The job of of each of them is to give commands to individual mobile sensors, until all the waypoints are covered.

		Input topics: target location estimation, the pose of each mobile sensor.

		Output behavior: generate new waypoints, and update the current waypoints of single_robot_controllers. 
		The single_robot_sensors will do their job to track the waypoints. This is in fact implementing an MPC algorithm. 

	"""
	def __init__(self, robot_names,pose_type_string,awake_freq=10):
		self.robot_names=robot_names
		self.awake_freq=awake_freq
		self.n_robots=len(robot_names)
		self.robot_names=robot_names

		rospy.init_node('multi_robot_controller',anonymous=True)
		self.listeners=[robot_listener(r,pose_type_string) for r in robot_names]

	def start(self):
		rate=rospy.Rate(self.awake_freq)

		while (not rospy.is_shutdown()):
			print('real_time_controlling')
			for l in self.listeners:
				if not(l.robot_pose==None):					
					l.robot_loc_stack.append(toxy(l.robot_pose))
					print(l.robot_name,l.robot_loc_stack[-1])

			rate.sleep()

		
		
if __name__ == '__main__':
	pose_type_string=prompt_pose_type_string()
	# print(pose_type_string)
	n_robots=3
	
	robot_names=['mobile_sensor_{}'.format(i) for i in range(n_robots)]
	
	mlt_controller=multi_robot_controller(robot_names,pose_type_string)	
	mlt_controller.start()
#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped,Pose, Twist
from std_msgs.msg import Float32MultiArray
import numpy as np
from functools import partial
import sys

from utils.RemotePCCodebase import prompt_pose_type_string,toxy,toyaw,stop_twist,get_twist
from robot_listener import robot_listener
from collections import deque
from utils.TurnAndGoTracking import TurnAndGo
from utils.LQRTracking import LQR_for_motion_mimicry

from geometry_msgs.msg import Twist

class single_robot_controller(object):
	"""
	 single_robot_controller

	 Interactions: receive waypoints from multi_robot_controller. Send out cmd_vel commands to mobile sensors.
	 Also receive pose from the mobile sensors.

	 Controller Kernel Algorithm: LQR for waypoint tracking.
	"""
	def __init__(self, robot_name,pose_type_string,awake_freq=10,kernel_algorithm='LQR',planning_dt=1):
		# Parameters
		self.robot_name=robot_name
		self.awake_freq=awake_freq
		self.kernel_algorithm=kernel_algorithm
		self.planning_dt = planning_dt

		# Data containers
		self.all_waypoints=None
		self.remaining_waypoints=None
		self.lqr_u=deque()
		self.curr_lqr_ind=0


		# ROS Setup
		rospy.init_node('single_robot_controller_{}'.format(self.robot_name))
		self.listener=robot_listener(robot_name,pose_type_string)
		self.waypoint_sub=rospy.Subscriber('/{}/waypoints'.format(self.robot_name),Float32MultiArray,self.waypoint_callback_)
		self.vel_pub=rospy.Publisher('/{}/cmd_vel'.format(self.robot_name),Twist,queue_size=10)
		
		self.Q=np.array([[10,0,0],[0,10,0],[0,0,1]])
		self.R_strength=1
		self.R = np.array([[10,0],[0,1]])
		

	def waypoint_callback_(self,data):
		a=self.all_waypoints=np.array(data.data).reshape(-1,2)
		if self.kernel_algorithm=='LQR':
			# print('waypoints received')
		
			loc=self.listener.robot_loc_stack[-1]
			yaw=self.listener.robot_yaw_stack[-1]
			curr_x = np.array([loc[0],loc[1],yaw])
			
			uhat,_,_ =LQR_for_motion_mimicry(self.all_waypoints,self.planning_dt,curr_x,Q=self.Q,R=self.R)
			self.lqr_u=uhat
			self.curr_lqr_ind=0
		self.remaining_waypoints=deque([self.all_waypoints[i,:] for i in range(len(self.all_waypoints))])


	def get_next_vel(self):
		vel_msg=stop_twist()
		if self.kernel_algorithm=='LQR':
			if len(self.lqr_u)>0 and self.curr_lqr_ind<len(self.lqr_u):
				[v,omega]=self.lqr_u[self.curr_lqr_ind]
				vel_msg=get_twist(v,omega)
				self.curr_lqr_ind+=1
		elif self.kernel_algorithm=='TurnAndGo':

			while len(self.remaining_waypoints)>0:	
				
				loc=self.listener.robot_loc_stack[-1]
				yaw=self.listener.robot_yaw_stack[-1]
				target_loc=self.remaining_waypoints[0]

				vel_msg=TurnAndGo(angular_vel_gain=6).get_twist(target_loc,loc,yaw)
				
				if vel_msg==stop_twist(): # For TurnAndGo, returning stop_twist means the current waypoint is arrived.
					self.remaining_waypoints.popleft()
				else: # If the returned vel_msg is not stop_twist, then exit the loop.
					break
					
		return vel_msg
	
	def update_remaining_waypoints(self):
		if not (self.remaining_waypoints is None):
			if len(self.remaining_waypoints)>0:
				pass	
			
	def start(self):
		rate=rospy.Rate(self.awake_freq)

		try:
			while not rospy.is_shutdown():
				if not(self.listener.robot_pose==None):					
					self.listener.robot_loc_stack.append(toxy(self.listener.robot_pose))
					self.listener.robot_yaw_stack.append(toyaw(self.listener.robot_pose))
				
				if not self.all_waypoints is None:
					vel_msg=self.get_next_vel()
					self.vel_pub.publish(vel_msg)
				# print("{} moving".format(self.robot_name))				
				rate.sleep()
		except:
			pass
		finally:
			print("{} Stoping".format(self.robot_name))
			self.vel_pub.publish(stop_twist())
	
		

if __name__ == '__main__':
	arguments = len(sys.argv) - 1
	
	kernel_algorithm = 'TurnAndGo'
	if arguments<=0:
		pose_type_string=prompt_pose_type_string()
		robot_no=input('The index for this robot is:')
	else:
		if arguments>=1:
			pose_type_string=sys.argv[1]
		if arguments>=2:
			robot_name=sys.argv[2]
		if arguments>=3:
			kernel_algorithm = sys.argv[3]

	awake_freq = 10
	planning_dt = 1
	controller=single_robot_controller(robot_name,pose_type_string,awake_freq=awake_freq,kernel_algorithm=kernel_algorithm,planning_dt=planning_dt)	
	controller.start()
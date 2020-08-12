#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped,Pose, Twist
from std_msgs.msg import Float32MultiArray,MultiArrayLayout,Bool,Float32
import numpy as np
from functools import partial
import sys

from utils.RemotePCCodebase import prompt_pose_type_string,toxy,get_sensor_names
from robot_listener import robot_listener

from utils.dLdp import analytic_dLdp,dLdp,L,dSdp
from utils.FIMPathPlanning import FIM_ascent_path_planning
from utils.ConcentricPathPlanning import concentric_path_planning
from utils.MutualSepPathPlanning import mutual_separation_path_planning
from utils.regions import Rect2D

BURGER_MAX_LIN_VEL = 0.22 * 0.8


class simple_waypoint_generator(object):
	"""

		multi_robot_controller

		Interacts with: a list of single_robot_controller nodes. 
			Each single_robot_controller has a sequence of waypoints to track. 
			The job of of each of them is to give commands to individual mobile sensors, until all the waypoints are covered.

		Input topics: target location estimation, the pose of each mobile sensor.

		Output behavior: generate new waypoints, and update the current waypoints of single_robot_controllers. 
		The single_robot_sensors will do their job to track the waypoints. This is in fact implementing an MPC algorithm. 

	"""
	def __init__(self, robot_names,pose_type_string,\
						xlim = (0,2.4),ylim = (0,4.5),\
						awake_freq=1,planning_dt=1):
		self.robot_names=robot_names
		self.awake_freq=awake_freq
		self.n_robots=len(robot_names)
		self.robot_names=robot_names
		self.xlim = xlim
		self.ylim = ylim
		
		# planning_timesteps should be at least (1/self.awake_freq)/planning_dt so that single_robot_controller
		# won't run out of waypoints between the awakenings of multi_robot_controller.
		self.planning_timesteps = 20 
		self.max_linear_speed = BURGER_MAX_LIN_VEL
		self.planning_dt = planning_dt
		self.epsilon=0.5

		# Data containers
		self.waypoints=None
	
		# ROS setup
		rospy.init_node('simple_waypoint_generator',anonymous=False)

		# Pose subscribers
		self.listeners=[robot_listener(r,pose_type_string) for r in robot_names]

		
		# Waypoint publishers
		self.waypoint_pub=dict()
		for name in self.robot_names:
			self.waypoint_pub[name]=rospy.Publisher('/{}/waypoints'.format(name),Float32MultiArray,queue_size=10)

	def get_waypoints(self):
		waypoints = []
		for i in range(self.planning_timesteps):
			w = [np.mean(self.xlim),i*BURGER_MAX_LIN_VEL*self.planning_dt +np.min(self.ylim)]
			waypoints.append([w for j in range(self.n_robots)])
		return np.array(waypoints)
	def start(self):
		rate=rospy.Rate(self.awake_freq)
		sim_time = 0
		while (not rospy.is_shutdown()):
			print('sim_time',sim_time)

			rate.sleep()

			print('real_time_controlling')

			# Update the robot pose informations.

			all_loc_received = True
			for l in self.listeners:
				if not(l.robot_pose==None):					
					l.robot_loc_stack.append(toxy(l.robot_pose))
				else:
					all_loc_received = False
			
			if not all_loc_received:
				continue
			ps=np.array([l.robot_loc_stack[-1] for l in self.listeners]).reshape(-1,2)
				# Start generating waypoints.
			self.waypoints=self.get_waypoints()
			self.waypoints=self.waypoints.reshape(-1,self.n_robots,2)
			

			for i in range(self.n_robots):
				out=Float32MultiArray()
				out.data=self.waypoints[:,i,:].ravel()
				self.waypoint_pub[self.robot_names[i]].publish(out)

			sim_time+=1/self.awake_freq
			
	
	
if __name__ == '__main__':
	
	print(sys.argv)

	arguments = len(sys.argv) - 1
	

	if arguments<=0:
		pose_type_string=prompt_pose_type_string()
	else:
		if arguments>=1:
			pose_type_string=sys.argv[1]
		
	robot_names=get_sensor_names()	

	n_robots = len(robot_names)
	

	g=simple_waypoint_generator(robot_names,\
										pose_type_string,\
										awake_freq= 10,\
										xlim=(0,2.4),\
										ylim=(0,4.5),\
										planning_dt = 1)	

	g.start()
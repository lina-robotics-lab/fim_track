#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped,Pose, Twist
from std_msgs.msg import Float32MultiArray,MultiArrayLayout,Bool,Float32
import numpy as np
from functools import partial
import sys

from utils.RemotePCCodebase import prompt_pose_type_string,toxy,get_sensor_names
from robot_listener import robot_listener
from StraightLineGenerator import StraightLineGenerator

from utils.dLdp import analytic_dLdp,dLdp,L,dSdp
from utils.FIMPathPlanning import FIM_ascent_path_planning
from utils.ConcentricPathPlanning import concentric_path_planning
from utils.MutualSepPathPlanning import mutual_separation_path_planning
from utils.StraightLinePathPlanning import straight_line_path_planning
from utils.regions import Rect2D

BURGER_MAX_LIN_VEL = 0.22 * 0.8


class multi_source_controller(object):
	"""

		multi_source_controller

		Interacts with: a list of single_source_controller nodes. 
			Each single_source_controller has a sequence of waypoints to track. 
			The job of of each of them is to give commands to individual mobile sensors, until all the waypoints are covered.

		Input topics: target location estimation, the pose of each mobile sensor.

		Output behavior: generate new waypoints, and update the current waypoints of single_source_controllers. 
		The single_source_sensors will do their job to track the waypoints. This is in fact implementing an MPC algorithm. 

	"""
	def __init__(self, source_names,pose_type_string,waypoint_generator,\
						awake_freq=1,xlim=(0.0,10.0),ylim=(0,10.0),planning_dt=1,epsilon=0.1):
		self.source_names=source_names
		self.awake_freq=awake_freq
		self.n_src=len(source_names)
		self.xlim = xlim
		self.ylim = ylim
		self.waypoint_generator = waypoint_generator

		# planning_timesteps should be at least (1/self.awake_freq)/planning_dt so that single_source_controller
		# won't run out of waypoints between the awakenings of multi_source_controller.
		self.planning_timesteps = 20 
		self.max_linear_speed = BURGER_MAX_LIN_VEL
		self.planning_dt = planning_dt
		self.epsilon=epsilon

		# Data containers
		self.waypoints=None
		# ROS setup
		rospy.init_node('source_movement_controller',anonymous=False)

		# Pose subscribers
		self.listeners=[robot_listener(r,pose_type_string) for r in source_names]

		# Waypoint publishers
		self.waypoint_pub=dict()
		for name in self.source_names:
			self.waypoint_pub[name]=rospy.Publisher('/{}/waypoints'.format(name),Float32MultiArray,queue_size=10)

	def start(self):
		rate=rospy.Rate(self.awake_freq)
		sim_time = 0
		while (not rospy.is_shutdown()):
			print('sim_time',sim_time)

			rate.sleep()
			# Update the source pose informations.

			for l in self.listeners:
				if not(l.robot_pose==None):					
					l.robot_loc_stack.append(toxy(l.robot_pose))

			qs=np.array([l.robot_loc_stack[-1] for l in self.listeners]).reshape(-1,2)

			print('The Source is Moving')				
			self.waypoints = self.waypoint_generator.get_waypoints(qs)


			self.waypoints=self.waypoints.reshape(-1,self.n_src,2)

			for i in range(self.n_src):
				out=Float32MultiArray()
				out.data=self.waypoints[:,i,:].ravel()
				self.waypoint_pub[self.source_names[i]].publish(out)
			sim_time+=1/self.awake_freq

	
if __name__ == '__main__':
	
	print(sys.argv)

	arguments = len(sys.argv) - 1
	

	if arguments<=0:
		pose_type_string=prompt_pose_type_string()
	else:
		if arguments>=1:
			pose_type_string=sys.argv[1]
		
	source_names=['target_0']

	n_src = len(source_names)

	waypoint_generator = StraightLineGenerator(pos_to_go = np.array([0,0]))

	src_controller=multi_source_controller(source_names,\
										pose_type_string,\
										waypoint_generator,\
										awake_freq= 10,\
									
										xlim=(0,np.inf),\
										ylim=(0,np.inf),\
										
										planning_dt = 1,\
										epsilon = 0.6)	

	src_controller.start()
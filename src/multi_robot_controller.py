#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped,Pose, Twist
from std_msgs.msg import Float32MultiArray,MultiArrayLayout,Bool
import numpy as np
from functools import partial
import sys

from RemotePCCodebase import prompt_pose_type_string,toxy,get_sensor_names
from robot_listener import robot_listener

from dLdp import analytic_dLdp,dLdp,L,dSdp
from FIMPathPlanning import FIM_ascent_path_planning
from ConcentricPathPlanning import concentric_path_planning
from regions import Rect2D

BURGER_MAX_LIN_VEL = 0.22 * 0.1


class multi_robot_controller(object):
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
						awake_freq=0.5,initial_movement_radius=0.5,initial_movement_time=5,xlim=(0.0,10.0),ylim=(0,10.0)):
		self.robot_names=robot_names
		self.awake_freq=awake_freq
		self.n_robots=len(robot_names)
		self.robot_names=robot_names
		self.xlim = xlim
		self.ylim = ylim

		# Path Planning Parameters
		self.initial_movement_finished = False
		self.initial_movement_radius = initial_movement_radius
		self.initial_movement_time=initial_movement_time

		self.planning_timesteps = 50
		self.max_linear_speed = BURGER_MAX_LIN_VEL
		self.planning_dt = 1
		self.epsilon=0.5

		# Data containers
		self.curr_est_locs=dict()
		self.waypoints=None
	
		# ROS setup
		rospy.init_node('multi_robot_controller',anonymous=False)

		# Pose subscribers
		self.listeners=[robot_listener(r,pose_type_string) for r in robot_names]

		# Estimated location subscribers
		self.est_loc_sub=dict()
		self.est_algs=[\
						'multi_lateration',\
						'intersection',\
						'ekf',\
						'pf',\
						'actual_loc'\
						]
		# self.est_algs=['pf']
		
		for alg in self.est_algs:
			self.est_loc_sub[alg]=rospy.Subscriber('/location_estimation/{}'.format(alg),Float32MultiArray, partial(self.est_loc_callback_,alg=alg))

		# Waypoint publishers
		self.waypoint_pub=dict()
		for name in self.robot_names:
			self.waypoint_pub[name]=rospy.Publisher('/{}/waypoints'.format(name),Float32MultiArray,queue_size=10)

		# The status flag indicating whether initial movement is over.
		self.initial_movement_pub = rospy.Publisher('/multi_robot_controller/initial_movement_finished',Bool,queue_size=10)
		
	def est_loc_callback_(self,data,alg):
		self.curr_est_locs[alg]=np.array(data.data)
	
	def get_est_loc(self):
		"""
			We will use a heuristic way to determine the estimated location based on the prediction from three candidate algorithms
		"""
		keys=self.curr_est_locs.keys()
		print('Current Target Location Estimates:',self.curr_est_locs)

		if 'actual_loc' in keys:
			return self.curr_est_locs['actual_loc']
		elif 'ekf' in keys:
			return self.curr_est_locs['ekf']
		elif 'pf' in keys:
		# elif False:
			return self.curr_est_locs['pf']
		elif 'multi_lateration' in keys:
			return self.curr_est_locs['multi_lateration']
		elif 'intersection' in keys:
			return self.curr_est_locs['intersection']
		else:
			return None

	def start(self):
		rate=rospy.Rate(self.awake_freq)
		sim_time = 0
		while (not rospy.is_shutdown()):

			rate.sleep()

			print('real_time_controlling')

			# Update the robot pose informations.

			all_loc_received = True
			for l in self.listeners:
				if not(l.robot_pose==None):					
					l.robot_loc_stack.append(toxy(l.robot_pose))
					print('Robotname:',l.robot_name,l.robot_loc_stack[-1])
				else:
					all_loc_received = False
					# print(l.robot_name,l.robot_loc_stack[-1])
			print(all_loc_received)
			
			if not all_loc_received:
				continue
			for alg, est in self.curr_est_locs.items():
				# print(alg,est)
				pass
			
			# Start generating waypoints.

			q=self.get_est_loc()
			if q is None:
				print('Not received any estimation locs')
				pass
			else:

				q=q.reshape(-1,2) # By default, this is using the estimation returned by ekf.

				ps=np.array([l.robot_loc_stack[-1] for l in self.listeners]).reshape(-1,2)


				C1s=[]
				C0s=[]
				ks=[]
				bs=[]
				for l in self.listeners:
					C1s.append(l.C1)
					C0s.append(l.C0)
					ks.append(l.k)
					bs.append(l.b)
				if None in C1s or None in C0s or None in ks or None in bs:
					print('Coefficients not fully yet received.')
				else:
					if not self.initial_movement_finished:		
						print('Performing Initial Movements')																								# R,ps,n_p,n_steps,max_linear_speed,dt,epsilon
						self.waypoints, radius_reached = concentric_path_planning(\
																		self.initial_movement_radius,ps,self.n_robots,\
																		self.planning_timesteps,\
																		self.max_linear_speed,\
																		self.planning_dt)
						self.initial_movement_finished = radius_reached or sim_time>=self.initial_movement_time
					else:
						print('Dynamic Tracking')
						# Feed in everything needed by the waypoint planner. 
						# After the initial movement is completed, we switch to FIM gradient ascent.
						
						# f_dLdp=partial(analytic_dLdp,C1s=C1s,C0s=C0s,ks=ks,bs=bs)
						
						f_dLdp=dLdp(C1s=C1s,C0s=C0s,ks=ks,bs=bs)
						
						# f_dLdp=dSdp(C1s=C1s,C0s=C0s,ks=ks,bs=bs)
						
						self.waypoints=FIM_ascent_path_planning(f_dLdp,q,ps,self.n_robots,\
																self.planning_timesteps,\
																self.max_linear_speed,\
																self.planning_dt,\
																self.epsilon,\
																Rect2D(self.xlim,self.ylim))
					
					# self.waypoints.shape should be (n_waypoints,n_sensors,space_dim)
					# For example, (31,3,2)
					self.initial_movement_pub.publish(self.initial_movement_finished)
					# print(self.initial_movement_finished)
					# print(self.waypoints.shape,self.initial_movement_finished)

					self.waypoints=self.waypoints.reshape(-1,self.n_robots,2)
					

					for i in range(self.n_robots):
						out=Float32MultiArray()
						out.data=self.waypoints[:,i,:].ravel()
						self.waypoint_pub[self.robot_names[i]].publish(out)

		sim_time+=1/self.awake_freq

		
		
if __name__ == '__main__':
	
	arguments = len(sys.argv) - 1
	

	if arguments<=0:
		pose_type_string=prompt_pose_type_string()
	else:
		if arguments>=1:
			pose_type_string=sys.argv[1]
		
	robot_names=get_sensor_names()	

	n_robots = len(robot_names)
	

	mlt_controller=multi_robot_controller(robot_names,\
										pose_type_string,\
										awake_freq=0.5,\
										initial_movement_radius=0.5,
										initial_movement_time=5,
										xlim=(0.0,10.0),\
										ylim=(0.0,10.0))	

	mlt_controller.start()
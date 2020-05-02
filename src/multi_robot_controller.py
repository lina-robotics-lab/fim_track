#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped,Pose, Twist
from std_msgs.msg import Float32MultiArray,MultiArrayLayout
import numpy as np
from functools import partial
import sys

from RemotePCCodebase import prompt_pose_type_string,toxy
from robot_listener import robot_listener

from dLdp import analytic_dLdp
from FIMPathPlanning import FIM_ascent_path_planning

BURGER_MAX_LIN_VEL = 0.22


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
	def __init__(self, robot_names,pose_type_string,awake_freq=1):
		self.robot_names=robot_names
		self.awake_freq=awake_freq
		self.n_robots=len(robot_names)
		self.robot_names=robot_names

		# Path Planning Parameters
		self.planning_timesteps = 50
		self.max_linear_speed = BURGER_MAX_LIN_VEL
		self.planning_dt = 0.5
		self.epsilon=0.1

		# Data containers
		self.curr_est_locs=dict()
		self.waypoints=None
	
		# ROS setup
		rospy.init_node('multi_robot_controller',anonymous=False)

		# Pose subscribers
		self.listeners=[robot_listener(r,pose_type_string) for r in robot_names]

		# Estimated location subscribers
		self.est_loc_sub=dict()
		self.est_algs=['multi_lateration','intersection','ekf']
		for alg in self.est_algs:
			self.est_loc_sub[alg]=rospy.Subscriber('/location_estimation/{}'.format(alg),Float32MultiArray, partial(self.est_loc_callback_,alg=alg))

		# Waypoint publishers
		self.waypoint_pub=dict()
		for name in self.robot_names:
			self.waypoint_pub[name]=rospy.Publisher('/{}/waypoints'.format(name),Float32MultiArray,queue_size=10)

		
	def est_loc_callback_(self,data,alg):
		self.curr_est_locs[alg]=np.array(data.data)
	
	def get_est_loc(self):
		"""
			By default, use ekf estimation as our estimated location.
		"""
		keys=self.curr_est_locs.keys()
		if 'ekf' in keys: 
			return self.curr_est_locs['ekf']
		elif 'intersection' in keys:
			return self.curr_est_locs['intersection']
		elif 'multi_lateration' in keys:
			return self.curr_est_locs['multi_lateration']
		else:
			return None

	def start(self):
		rate=rospy.Rate(self.awake_freq)

		while (not rospy.is_shutdown()):
			print('real_time_controlling')

			# Update the robot pose informations.

			for l in self.listeners:
				if not(l.robot_pose==None):					
					l.robot_loc_stack.append(toxy(l.robot_pose))
					# print(l.robot_name,l.robot_loc_stack[-1])
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
					# Feed in everything needed by the waypoint planner. 
					# By default we use FIM gradient ascent.
					df_dLdp=partial(analytic_dLdp,C1s=C1s,C0s=C0s,ks=ks,bs=bs)
					self.waypoints=FIM_ascent_path_planning(df_dLdp,q,ps,self.n_robots,self.planning_timesteps,self.max_linear_speed,self.planning_dt,self.epsilon)
					print(self.waypoints.shape)
					self.waypoints=self.waypoints.reshape(-1,self.n_robots,2)
					

					for i in range(self.n_robots):
						out=Float32MultiArray()
						out.data=self.waypoints[:,i,:].ravel()
						self.waypoint_pub[self.robot_names[i]].publish(out)

					"""
						To do: implement the single robot controllers, and feed them with the waypoints!
					"""

			rate.sleep()

		
		
if __name__ == '__main__':
	
	arguments = len(sys.argv) - 1
	

	if arguments<=0:
		pose_type_string=prompt_pose_type_string()
		n_robots=int(input('The number of mobile sensors:'))
	else:
		if arguments>=1:
			pose_type_string=sys.argv[1]
		if arguments>=2:
			n_robots=int(sys.argv[2])
			
	robot_names=['mobile_sensor_{}'.format(i) for i in range(n_robots)]

	mlt_controller=multi_robot_controller(robot_names,pose_type_string)	
	mlt_controller.start()
#!/usr/bin/env python3
import rospy
import argparse
from geometry_msgs.msg import PoseStamped,Pose, Twist
from std_msgs.msg import Float32MultiArray,Float32,Bool
from turtlesim.msg import Pose as tPose
from nav_msgs.msg import Odometry

import numpy as np
import sys
from RemotePCCodebase import *
from DynamicFilters import getDynamicFilter
from robot_listener import robot_listener
from regions import Rect2D

class location_estimation:
	def __init__(self,robot_names,pose_type_string,qhint=None,awake_freq=10,target_name=None,xlim=(0.0,2.4),ylim=(0,4.5)):
		"""
			pose_type_string is one in ["turtlesimPose", "Pose", "Odom", "optitrack"]
		"""
		rospy.init_node('location_estimation',anonymous=True)

		robot_coef_dict=dict({r:np.zeros((4,)) for r in robot_names})
		
		self.xlim = xlim
		self.ylim = ylim		
		self.n_robots=len(robot_names)
		self.robot_names=robot_names
		self.listeners=[robot_listener(r,pose_type_string) for r in robot_names]

		self.target_name = target_name
		self.true_target_loc=[]
		self.estimated_locs=[]
		self.scalar_readings = dict()
		
		self.awake_freq=awake_freq		

		self.initial_movement_finished=False
		rospy.Subscriber('/multi_robot_controller/initial_movement_finished',Bool,self.initial_movement_callback_)
		
		# Prepare the publishers of location estimation, one for each estimation algorithm.
		self.estimation_pub=dict()
		self.dynamic_filter_algs = [\
									'ekf',\
									'pf']
		self.dynamic_filters=dict({alg:None for alg in self.dynamic_filter_algs})

		self.algs=['multi_lateration','intersection']
		self.algs.extend(self.dynamic_filter_algs)

		self.target_listener = None
		if target_name!=None:
				self.target_listener = robot_listener(target_name,pose_type_string)
				self.algs.append('actual_loc')
		
		for alg in self.algs:
			self.estimation_pub[alg]=rospy.Publisher('location_estimation/{}'.format(alg),Float32MultiArray,queue_size=10)

		# Prepare the scalar reading publisher, useful for hill climbing algorithms.
		self.scalar_readings_pub = dict()
		for name in self.robot_names:
			self.scalar_readings_pub[name] = rospy.Publisher('{}/scalar_readings'.format(name),Float32,queue_size=10)
	
	
	def localize_target(self,look_back=30):
		# Step 1: get rhat estimation from each of the listeners
		rhats=[]
		sensor_locs=[]

		# Thest two are for self.dynamic_filter. It does not require lookback but only needs the latest sensor loc and readings.
		latest_scalar_readings=[]
		latest_sensor_locs=[]
		scalar_readings_dict = dict()

		for l in self.listeners:
			if l.k!=None and len(l.light_reading_stack)>0:
						
				lookback_len=np.min([look_back,len(l.light_reading_stack),len(l.robot_loc_stack)])

				meas=top_n_mean(np.array(l.light_reading_stack[-lookback_len:]),2)

				rh=rhat(meas,l.C1,l.C0,l.k,l.b)

				# Handle some nan rhat issue by interpolation.
				# if np.any(np.isnan(rh)):
				# 	print(l.robot_name)
				rh[np.isnan(rh)]=np.mean(rh[np.logical_not(np.isnan(rh))])
	
				rhats.append(rh)

				loc=np.array(l.robot_loc_stack[-lookback_len:]).reshape(-2,2)
				sensor_locs.append(loc)
				l.rhats.append(rh)

				# Thest two are for self.dynamic_filter.
				latest_scalar_readings.append(meas[-1])
				scalar_readings_dict[l.robot_name]=meas[-1]
				latest_sensor_locs.append(loc[-1,:])
			else:
				pass
	
		if len(rhats)>0:
			estimates=dict()
			estimates['multi_lateration']=multi_lateration_from_rhat(np.vstack(sensor_locs),np.hstack(rhats).ravel())
			
			qhint = estimates['multi_lateration']
			
			estimates['intersection']=intersection_localization(np.vstack(sensor_locs),np.hstack(rhats).ravel(),qhint)
			
			for key,dynamic_filter in self.dynamic_filters.items():
				if not dynamic_filter is None:
					estimates[key]=dynamic_filter.update_and_estimate_loc(np.array(latest_sensor_locs),np.array(latest_scalar_readings))
			
			# Perform the estimation projection
			box = Rect2D(self.xlim,self.ylim)
			for key,val in estimates.items():
				if not val is None:
					estimates[key]=box.project_point(val)
			

			if not self.target_listener is None:
				actual_loc = self.target_listener.robot_pose
				if not actual_loc is None:
					estimates['actual_loc']=toxy(actual_loc)
			
			return estimates, scalar_readings_dict
		else:
			return None
	
	def initial_movement_callback_(self,finished):
		self.initial_movement_finished = finished.data
	
	
	def start(self,target_name=None,save_data=False,trail_num=0):
		
		
		rate=rospy.Rate(self.awake_freq)

		NUM_TARGET=1
				

		while (not rospy.is_shutdown()):
			

			# Gather the latest readings from listeners
			for l in self.listeners:
				# print(l.light_readings,l.robot_name,l.robot_pose)
				if not(l.robot_pose==None or l.light_readings==None):					
					# print('name:',l.robot_name)
					l.robot_loc_stack.append(toxy(l.robot_pose))
					l.light_reading_stack.append(np.array(l.light_readings))


			'''
			The Magic to here: real-time localization algorithm.
			Should be called after the location & light reading update is done.
			'''
			result=self.localize_target()
			if not result is None:
				est_loc,self.scalar_readings=result
				C1s=[]
				C0s=[]
				ks=[]
				bs=[]
				for l in self.listeners:
					C1s.append(l.C1)
					C0s.append(l.C0)
					ks.append(l.k)
					bs.append(l.b)					
				
				# Set the initial guess of the dynamic filter to be the current est_loc.
				# It comes from multi-lateration or intersection method.
				initial_guess = est_loc['multi_lateration'].reshape(2,)

				# Initialize the dynamic filter if the initial movements from the sensors are finished.
				if self.initial_movement_finished:
					for filter_alg in self.dynamic_filter_algs:
						if self.dynamic_filters[filter_alg] is None:			
							self.dynamic_filters[filter_alg]=getDynamicFilter(self.n_robots,NUM_TARGET,C1s,C0s,ks,bs,initial_guess=initial_guess,filterType=filter_alg)
							if not self.dynamic_filters[filter_alg] is None:
								print('{} initialized'.format(filter_alg))
								# print('initial_movement_finished',self.initial_movement_finished)


				self.estimated_locs.append(est_loc)
				# print('\n Estimation of target location')
				
				# Publish Estimations
				for alg,est in est_loc.items():		
					if not est is None:	
						out=Float32MultiArray()
						out.data=est
						self.estimation_pub[alg].publish(out)
				# Publish Scalar Readings
				for robot_name,val in self.scalar_readings.items():
					out=Float32()
					out.data = val
					self.scalar_readings_pub[robot_name].publish(out)

			if not self.target_name is None and not self.target_listener.robot_pose is None:
				self.true_target_loc.append(toxy(self.target_listener.robot_pose))
			rate.sleep()

				
			
		
if __name__=='__main__':
	

	arguments = len(sys.argv) - 1

	if arguments<=0:
		pose_type_string=prompt_pose_type_string()
	else:
		if arguments>=1:
			pose_type_string=sys.argv[1]
		
	robot_names=get_sensor_names()		
	n_robots = len(robot_names)
	
	target_name='target_0'
	# target_name=None

	
	qhint=np.array([0.0,0.0])
	# qhint=None
	
	le=location_estimation(robot_names,pose_type_string,qhint=qhint,awake_freq=10,target_name=target_name)
	le.start(target_name=target_name,trail_num=7)


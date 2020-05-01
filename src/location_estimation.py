#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped,Pose, Twist
from std_msgs.msg import Float32MultiArray
import numpy as np
import sys
from RemotePCCodebase import *
from DynamicFilters import getDynamicFilter

from turtlesim.msg import Pose as tPose
from nav_msgs.msg import Odometry



class robot_listener:
	''' Robot location and light_reading listener+data container.'''
	def __init__(self,robot_name,pose_type_string):
		"""
			pose_type_string is one in ["turtlesimPose", "Pose", "Odom", "optitrack"]
		"""
		self.robot_name=robot_name
		print('initializing {} listener'.format(robot_name))
		
		
		self.pose_type,self.rpose_topic=get_pose_type_and_topic(pose_type_string,robot_name)

		
		self.light_topic="/{}/sensor_readings".format(robot_name)
		self.coefs_topic="/{}/sensor_coefs".format(robot_name)
		self.robot_pose=None
		self.light_readings=None

		self.robot_loc_stack=[]
		self.light_reading_stack=[]
		self.rhats=[]

		self.C1=None
		self.C0=None
		self.k=None
		self.b=None



	def sensor_coef_callback_(self,data):
		coefs=data.data
		self.C1,self.C0,self.k,self.b=coefs

	def robot_pose_callback_(self,data):
		self.robot_pose=data

	def light_callback_(self,data):

		self.light_readings=data.data


class location_estimation:
	def __init__(self,robot_names,pose_type_string,awake_freq=10,localization_alg='multi_lateration',qhint=np.array([0,0]),dynamic_filter_type='ekf'):
		"""
			pose_type_string is one in ["turtlesimPose", "Pose", "Odom", "optitrack"]
		"""
		rospy.init_node('location_estimation',anonymous=True)

		robot_coef_dict=dict({r:np.zeros((4,)) for r in robot_names})
		
		
		self.n_robots=len(robot_names)
		self.robot_names=robot_names
		self.listeners=[robot_listener(r,pose_type_string) for r in robot_names]

		
		self.target_pose=None
		self.true_target_loc=[]
		self.estimated_locs=[]

		self.awake_freq=awake_freq		

		self.localization_alg=localization_alg

		self.qhint=qhint

		self.dynamic_filter=None
		self.dynamic_filter_type=dynamic_filter_type

		for l in self.listeners:
			rospy.Subscriber(l.rpose_topic, l.pose_type, l.robot_pose_callback_)
			rospy.Subscriber(l.light_topic, Float32MultiArray, l.light_callback_)
			rospy.Subscriber(l.coefs_topic, Float32MultiArray, l.sensor_coef_callback_)
			
		if target_name!=None:
			rospy.Subscriber('/vrpn_client_node/{}/pose'.format(target_name),PoseStamped,self.target_pose_callback_)

		
		# Prepare the publishers of location estimation, one for each estimation algorithm.
		self.estimation_pub=dict()
		self.algs=['multi_lateration','intersection','ekf']
		
		for alg in self.algs:
			self.estimation_pub[alg]=rospy.Publisher('location_estimation/{}'.format(alg),Float32MultiArray,queue_size=10)

	
	
	def localize_target(self,look_back=30):
		# Step 1: get rhat estimation from each of the listeners
		rhats=[]
		sensor_locs=[]

		# Thest two are for self.dynamic_filter. It does not require lookback but only needs the latest sensor loc and readings.
		latest_scalar_readings=[]
		latest_sensor_locs=[]

		for l in self.listeners:
			if l.k!=None and len(l.light_reading_stack)>0:
						
				lookback_len=np.min([look_back,len(l.light_reading_stack),len(l.robot_loc_stack)])

				meas=top_n_mean(np.array(l.light_reading_stack[-lookback_len:]),2)

				rh=rhat(meas,l.C1,l.C0,l.k,l.b)
				# print(l.C1,l.C0,l.k,l.b)

				rhats.append(rh)

				loc=np.array(l.robot_loc_stack[-lookback_len:]).reshape(-2,2)
				sensor_locs.append(loc)
				l.rhats.append(rh)

				# Thest two are for self.dynamic_filter.
				latest_scalar_readings.append(meas[-1])
				latest_sensor_locs.append(loc[-1,:])
			else:
				print(l.robot_name,l.k,len(l.light_reading_stack))
				# print(l.robot_name,rh[0])	
		# print('rh',rhats)
		if len(rhats)>0:
			estimates=dict()
			estimates['multi_lateration']=multi_lateration_from_rhat(np.vstack(sensor_locs),np.hstack(rhats).ravel())
			estimates['intersection']=intersection_localization(np.vstack(sensor_locs),np.hstack(rhats).ravel(),self.qhint)
			if not self.dynamic_filter is None:
				estimates['ekf']=self.dynamic_filter.update_and_estimate_loc(np.array(latest_sensor_locs),np.array(latest_scalar_readings))
			return estimates
		else:
			return None

	def target_pose_callback_(self,data):
		# print(data)
		self.target_pose=data

	def real_time_localization(self,target_name=None,save_data=True,trail_num=0):
		
		
		rate=rospy.Rate(self.awake_freq)

		while (not rospy.is_shutdown()):

			# Initialize the dynamic filter if it is not yet done.
			if self.dynamic_filter is None:
				C1s=[]
				C0s=[]
				ks=[]
				bs=[]
				for l in self.listeners:
					C1s.append(l.C1)
					C0s.append(l.C0)
					ks.append(l.k)
					bs.append(l.b)

				NUM_TARGET=1
				
				self.dynamic_filter=getDynamicFilter(self.n_robots,NUM_TARGET,C1s,C0s,ks,bs,initial_guess=self.qhint)
				if not self.dynamic_filter is None:
					print('dynamic_filter initialized')


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
			est_loc=self.localize_target()
			if not est_loc is None:
				self.estimated_locs.append(est_loc)
				print('\n Estimation of target location')
				
				for alg,est in est_loc.items():
					print('Estimated by {}:{}'.format(alg,est))
					
					out=Float32MultiArray()
					out.data=est
					self.estimation_pub[alg].publish(out)
			
			if target_name!=None and self.target_pose!=None:
				self.true_target_loc.append(pose2xz(self.target_pose))
			rate.sleep()

		if save_data:
			np.savetxt('estimated_locs_{}.txt'.format(trail_num),np.array(self.estimated_locs),delimiter=',')
			for l in self.listeners:
				np.savetxt('sensor_locs_{}_{}.txt'.format(l.robot_name,trail_num),np.array(l.robot_loc_stack),delimiter=',')
				np.savetxt('rhats_{}_{}.txt'.format(l.robot_name,trail_num),np.hstack(l.rhats).ravel(),delimiter=',')
				np.savetxt('light_readings_{}_{}.txt'.format(l.robot_name,trail_num),l.light_reading_stack,delimiter=',')
				np.savetxt('coefs_{}_{}.txt'.format(l.robot_name,trail_num),[l.C1,l.C0,l.k,l.b],delimiter=',')
				
			if target_name!=None:
				np.savetxt('true_target_loc_{}.txt'.format(trail_num),np.array(self.true_target_loc),delimiter=',')
		
		
if __name__=='__main__':
	platform_2_pose_types=dict()
	platform_2_pose_types['s']="turtlesimPose"
	platform_2_pose_types['g']='Odom'
	platform_2_pose_types['t']='Pose'
	platform_2_pose_types['o']='optitrack'

	platform=input("Please indicate the platform of your experiment.\n s => turtlesim\n g => Gazebo\n t => Real Robots Turtlebot3 \n o => Optitrack:")
	pose_type_string=platform_2_pose_types[platform]

	arguments = len(sys.argv) - 1

	robot_names=['mobile_sensor_{}'.format(i) for i in range(3)]
	# for i in range(1,arguments+1,1):
	#	robot_names.append(sys.argv[i])

	# target_name='target_0'
	target_name=None

	localization_alg='ekf'
	qhint=np.array([5.0,5.0])
	
	le=location_estimation(robot_names,pose_type_string,localization_alg=localization_alg,qhint=qhint)
	le.real_time_localization(target_name=target_name,trail_num=7)


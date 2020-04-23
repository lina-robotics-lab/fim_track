#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped,Pose, Twist
from std_msgs.msg import Float32MultiArray
import numpy as np
import sys
from RemotePCCodebase import *

from turtlesim.msg import Pose as tPose
from nav_msgs.msg import Odometry



class robot_listener:
	''' Robot location and light_reading listener+data container.'''
	def __init__(self,robot_name,pose_type='turtlesimPose'):
		self.robot_name=robot_name
		print('initializing {} listener'.format(robot_name))
		
		
		self.pose_type,self.rpose_topic=get_pose_type_and_topic(pose_type,robot_name)

		
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
	def __init__(self,robot_names,awake_freq=10,localization_alg='multi_lateration',qhint=np.array([0,0]),pose_type='turtlesimPose'):

		rospy.init_node('location_estimation',anonymous=True)

		robot_coef_dict=dict({r:np.zeros((4,)) for r in robot_names})
		
		
		self.n_robots=len(robot_names)
		self.robot_names=robot_names
		self.listeners=[robot_listener(r) for r in robot_names]

		
		self.target_pose=None
		self.true_target_loc=[]
		self.estimated_locs=[]

		self.awake_freq=awake_freq		

		self.localization_alg=localization_alg

		self.qhint=qhint

		for l in self.listeners:
			rospy.Subscriber(l.rpose_topic, l.pose_type, l.robot_pose_callback_)
			rospy.Subscriber(l.light_topic, Float32MultiArray, l.light_callback_)
			rospy.Subscriber(l.coefs_topic, Float32MultiArray, l.sensor_coef_callback_)
			
		if target_name!=None:
			rospy.Subscriber('/vrpn_client_node/{}/pose'.format(target_name),PoseStamped,self.target_pose_callback_)

		
	
	
	def localize_target(self,look_back=30):
		# Step 1: get rhat estimation from each of the listeners
		rhats=[]
		sensor_locs=[]

		for l in self.listeners:
			if l.k!=None and len(l.light_reading_stack)>0:
						
				lookback_len=np.min([look_back,len(l.light_reading_stack),len(l.robot_loc_stack)])

				scalar_readings=top_n_mean(np.array(l.light_reading_stack[-lookback_len:]),2)

				rh=rhat(scalar_readings,l.C1,l.C0,l.k,l.b)
				# print(l.C1,l.C0,l.k,l.b)

				rhats.append(rh)

				loc=np.array(l.robot_loc_stack[-lookback_len:]).reshape(-2,2)
				sensor_locs.append(loc)
				l.rhats.append(rh)

				# print(l.robot_name,rh[0])	

		if len(rhats)>0:
			# print('rh',np.hstack(rhats).ravel().shape,'loc',np.vstack(sensor_locs).shape)
			if self.localization_alg=='multi_lateration':
				return multi_lateration_rhat(np.vstack(sensor_locs),np.hstack(rhats).ravel())
			elif self.localization_alg=='intersection':
				return intersection_localization(np.vstack(sensor_locs),np.hstack(rhats).ravel(),self.qhint)
		return None

	def target_pose_callback_(self,data):
		# print(data)
		self.target_pose=data

	def real_time_localization(self,target_name=None,save_data=True,trail_num=0):
		
		
		rate=rospy.Rate(self.awake_freq)

		while (not rospy.is_shutdown()):
			# Gather the latest readings from listeners
			for l in self.listeners:
				# print(l.light_readings,l.robot_name)
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
				print(est_loc)
			
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
	arguments = len(sys.argv) - 1

	robot_names=['mobile_sensor_{}'.format(i) for i in range(3)]
	# for i in range(1,arguments+1,1):
	#	robot_names.append(sys.argv[i])

	target_name='target_0'
	# target_name=None

	localization_alg='intersection'
	qhint=np.array([10.0,10.0])
	
	le=location_estimation(robot_names,localization_alg=localization_alg,qhint=qhint)
	le.real_time_localization(target_name=target_name,trail_num=7)


#!/usr/bin/env python3
import rospy
import sys
from std_msgs.msg import Float32MultiArray,Float32
from geometry_msgs.msg import Pose
from turtlesim.msg import Pose as tPose
from nav_msgs.msg import Odometry
import numpy as np
from functools import partial

from RemotePCCodebase import yaw_from_odom,xy_from_odom

class virtual_sensor(object):
	"""
	Remember to use virtual_sensor.py (not just virtual_sensor) to refer to this package.

	The virtual sensor used in simulations. 
	"""
	def __init__(self, robot_namespace,target_namespaces, publish_rate=10,pose_type='turtlesimPose', num_sensors=8):
		
		# Parameters ##########################################
		self.robot_namespace=robot_namespace
		self.target_namespaces=target_namespaces
		
		self.num_sensors=num_sensors

		if pose_type=='turtlesimPose':
			self.pose_type=tPose
			self.pose_topic='pose'
		elif pose_type=='Pose':
			self.pose_type=Pose
			self.pose_topic='pose'
		elif pose_type=='Odom':
			self.pose_type=Odometry
			self.pose_topic='odom'


		self.C1=0
		self.C0=2
		self.b=-2
		self.noise_std=1e-1
		self.relative_theta=np.linspace(0,1,num_sensors)*np.pi



		# Local Data Containers #####################################
		
		self.robot_position=None
		self.robot_angle=None
		self.robot_pose_buffer=None

		self.target_positions=dict({ns:None for ns in target_namespaces})
		self.target_angles=dict({ns:None for ns in target_namespaces})
		self.target_poses_buffer=dict({ns:None for ns in target_namespaces})
	
		self.raw_light_strengths=dict({ns:None for ns in target_namespaces})	
		self.raw_light_strengths_buffer=dict({ns:None for ns in target_namespaces})	

		self.sensor_readings=np.zeros(num_sensors)
		

		# ROS configurations ####################################

		rospy.init_node("virtual sensor {}".format(self.robot_namespace),anonymous=False)
		self.rate=rospy.Rate(publish_rate)

		# Subscribers ########################################
		
		for target in target_namespaces:
			
			light_sub=rospy.Subscriber('{}/raw_light_strength'.format(target),Float32,partial(self.raw_light_strength_callback,namespace=target))			
			sub=rospy.Subscriber('{}/{}'.format(target,self.pose_topic),self.pose_type,partial(self.target_pose_callback,namespace=target))
	
		robot_sub=rospy.Subscriber('{}/{}'.format(self.robot_namespace,self.pose_topic),self.pose_type,self.robot_pose_callback)
		
	
		# Publisher #########################################
		# The topic name sensor_readings is consistent with the rest of the package.
		self.light_reading_pub=rospy.Publisher('{}/sensor_readings'.format(self.robot_namespace),Float32MultiArray,queue_size=10)	
		
	def raw_light_strength_callback(self,data,namespace=''):
		# print('lights call back')
		self.raw_light_strengths_buffer=data

	def target_pose_callback(self,pose,namespace=''):
		# print('target pose call back',namespace)
		self.target_poses_buffer[namespace]=pose
		
	def robot_pose_callback(self,pose):
		# print('robot pose callback')
		self.robot_pose_buffer=pose
	
	def tPose2xy(self,data):
		return np.array([data.x,data.y])
	def tPose2yaw(self,data):
		return data.theta
	

	def publish_readings(self):

		# Make sure the buffers are populated

		
		target_filled=True
		for target in self.target_namespaces:
			if self.target_poses_buffer[target] is None:
				target_filled=False
				break

			
		# print(target_filled,(not self.robot_pose_buffer is None) , (not self.raw_light_strengths_buffer is None))
		if (not self.robot_pose_buffer is None) and (not self.raw_light_strengths_buffer is None) and target_filled:	
			# print('can publish')
			self.raw_light_strengths=self.raw_light_strengths_buffer
			
			if self.pose_type is tPose:
				toxy=self.tPose2xy
				toyaw=self.tPose2yaw
			elif self.pose_type is Odometry:
				toxy=self.xy_from_odom
				toyaw=self.yaw_from_odom

			self.robot_position=toxy(self.robot_pose_buffer)
			self.robot_angle=toyaw(self.robot_pose_buffer)
				
			for target in self.target_namespaces:
				self.target_positions[target]=toxy(self.target_poses_buffer[target])
				self.target_angles[target]=toyaw(self.target_poses_buffer[target])
			# print(self.robot_position,self.robot_angle)
			# The calculations are done below

			out=Float32MultiArray()
			out.data=self.sensor_readings
			# out.data=list(self.sensor_readings)
			self.light_reading_pub.publish(out)

if __name__ == '__main__':
	if len(sys.argv)<=1:
		print('Please specify the namespace of the virtual sensor!')
	elif len(sys.argv)<=2:
		print('Please specify the namespace of the virtual sensor and virtual lights!')
	else:
		pose_type=sys.argv[1]
		robot_namespace=sys.argv[2]
		target_namespaces=sys.argv[3:-2]

		# print(robot_namespace)
		# print(target_namespaces)

		v=virtual_sensor(robot_namespace,target_namespaces,pose_type=pose_type)
		while not rospy.is_shutdown():
			try:
				v.publish_readings()
				v.rate.sleep()
			except rospy.exceptions.ROSInterruptException:
				# After Ctrl+C is pressed.
				pass

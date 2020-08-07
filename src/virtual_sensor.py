#!/usr/bin/env python3
import rospy
import sys
from std_msgs.msg import Float32MultiArray,Float32
from geometry_msgs.msg import Pose
from turtlesim.msg import Pose as tPose
from nav_msgs.msg import Odometry
import numpy as np
from functools import partial

from utils.RemotePCCodebase import *

class virtual_sensor(object):
	"""
	Remember to use virtual_sensor.py (not just virtual_sensor) to refer to this package.

	The virtual sensor used in simulations. 
	"""
	def __init__(self, robot_namespace,target_namespaces, publish_rate=10,pose_type_string='turtlesimPose', num_sensors=8, output_stype='uniform'):
		
		"""
			pose_type_string is one in ["turtlesimPose", "Pose", "Odom", "optitrack"]

			output_style is one in ["uniform","rotational"]
		"""

		# Parameters ##########################################
		self.robot_namespace=robot_namespace
		self.target_namespaces=target_namespaces
		
		self.num_sensors=num_sensors

		self.pose_type,_=get_pose_type_and_topic(pose_type_string,'')

		self.C1=0
		self.C0=0
		self.b=-2
		self.r=0.1
		self.noise_std=rospy.get_param('noise_level')
		self.relative_theta=np.linspace(0,1,num_sensors)*np.pi

		self.output_stype=output_stype


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

			_,topic=get_pose_type_and_topic(pose_type_string,target)
			sub=rospy.Subscriber(topic,self.pose_type,partial(self.target_pose_callback,namespace=target))
	
		_,topic = get_pose_type_and_topic(pose_type_string,robot_namespace)
		robot_sub=rospy.Subscriber( topic,self.pose_type,self.robot_pose_callback)
		
	
		# Publisher #########################################
		# The topic name sensor_readings is consistent with the rest of the package.
		self.light_reading_pub=rospy.Publisher('{}/sensor_readings'.format(self.robot_namespace),Float32MultiArray,queue_size=10)	
		self.sensor_coefs_pub=rospy.Publisher('{}/sensor_coefs'.format(self.robot_namespace),Float32MultiArray,queue_size=10)
		
	def raw_light_strength_callback(self,data,namespace=''):
		# print('lights call back')
		self.raw_light_strengths_buffer[namespace]=data.data

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
	
	def calculate_influence(self,target_name):
		
		if self.output_stype=='uniform':
			"""
				Output the same reading for every light sensor.
			"""
			# The displacement between the target and the robot
			q2p=self.target_positions[target_name]-self.robot_position

			d=np.linalg.norm(q2p)

			k=self.raw_light_strengths[target_name]
			
			l = np.ones(self.num_sensors)*(k*(d-self.C1)**self.b)

		elif self.output_stype=='rotational':
			"""
				Advanced. Taking into account the effect of rotation on light sensor
				readings, output a different reading for each light sensor.
			"""
			# l[i] denotes the influence of the target on the ith sensor.
			l=np.zeros(len(self.target_namespaces))
			
			# The displacement between the target and the robot
			q2p=self.target_positions[target_name]-self.robot_position

			# atan2(y,x) returns the angle formed by (x,y) and x axis, ranges in [-pi,pi].
			phi=np.arctan2(q2p[1],q2p[0])

			# psi has shape (self.num_sensors,), the angle formed by sensor-CM of robot-target.
			psi=self.relative_theta+self.robot_angle-phi

			# d is the distances of individual sensors to the target. The individual sensors are
			# located at r distance from the center of the robot(mobile sensor).
			# The formula is the cosine rule
			# d should have shape (self.num_sensors,)
			d=np.sqrt(self.r**2+np.linalg.norm(q2p)**2-2*self.r*np.linalg.norm(q2p)*np.cos(psi))

			# The measurement model, taking into account the facing angle between the sensor and the light direction.
			# The sensors in the shawdow will only receive background noise.
			# l should have shape (self.num_sensors,)
			k=self.raw_light_strengths[target_name]
			l = np.max(np.array([(k*(d-self.C1)**self.b )*np.cos(psi),np.zeros(self.num_sensors)]),axis=0)
			
		return l
	def publish_coefs(self):
		out=Float32MultiArray()
		if not self.raw_light_strengths_buffer is None:
			if self.output_stype=='uniform':
				ks=self.raw_light_strengths_buffer
				if len(ks)>1:
					out.data=np.array([[self.C1,self.C0,k,self.b] for _,k in ks.items()])
				elif len(ks)==1:
					out.data=np.array([self.C1,self.C0,ks[self.target_namespaces[0]],self.b])
				self.sensor_coefs_pub.publish(out)
			else:
				print("output_stype other than uniform is not yet supported, since it requires coefficient calibration.")
				pass

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

			
			self.robot_position=toxy(self.robot_pose_buffer)
			self.robot_angle=toyaw(self.robot_pose_buffer)
				
			for target in self.target_namespaces:
				self.target_positions[target]=toxy(self.target_poses_buffer[target])
				self.target_angles[target]=toyaw(self.target_poses_buffer[target])
			# print(self.robot_position,self.robot_angle)
			
			# The calculations are done below

			# out=Float32MultiArray()

			# out.data=self.sensor_readings

			# out.data=list(self.sensor_readings)
			influences=[]
			for target in self.target_namespaces:
				infl=self.calculate_influence(target)
				influences.append(infl)
			
			# The superposition principle: the influence(light strength) of each target(light source)
			# on each sensor sums to the readings on each sensor
			# plus some iid white noise.
			# We do not truncate the influences to be above zero here.
			
			influences = np.array(influences) # shape = (num_targets,num_sensors)
			self.sensor_readings=np.sum(influences,axis=0) # shape = (num_senosrs,)
			self.sensor_readings+=np.random.randn(self.num_sensors)*self.noise_std # Add the iid white noise.


			out=Float32MultiArray()
			out.data=self.sensor_readings
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

		v=virtual_sensor(robot_namespace,target_namespaces,pose_type_string=pose_type)
		while not rospy.is_shutdown():
			try:
				v.publish_coefs()
				v.publish_readings()
				v.rate.sleep()
			except rospy.exceptions.ROSInterruptException:
				# After Ctrl+C is pressed.
				pass

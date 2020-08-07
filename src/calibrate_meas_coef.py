#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped,Pose, Twist
from std_msgs.msg import Float32MultiArray
import numpy as np
import sys
from spin_and_collect import spin_and_collect
from utils.RemotePCCodebase import calibrate_meas_coef as cc
from utils.RemotePCCodebase import pose2xz

class calibrate_meas_coef:
	def __init__(self):
		self.robot_pose=None
		self.target_pose=None
		self.light_readings=None

		self.robot_loc_stack=[]
		self.target_loc_stack=[]
		self.light_reading_stack=[]

		self.awake_freq=10
		
		
	def robot_pose_callback_(self,data):
		# print(data.pose)
		self.robot_pose=data.pose
		
	def target_pose_callback_(self,data):
		self.target_pose=data.pose

	def light_callback_(self,data):
		print(data.data)
		self.light_readings=data.data

	
	def record_and_calibrate(self,robot_namespace,target_namespace,save_data=False,fit_type='light_readings'):
		rospy.init_node('calibrate_meas_coef',anonymous=True)
		
		rpose_topic="/vrpn_client_node/{}/pose".format(robot_namespace)
		tpose_topic="/vrpn_client_node/{}/pose".format(target_namespace)

		robot_pose=rospy.Subscriber(rpose_topic, PoseStamped, self.robot_pose_callback_)
		target_pose=rospy.Subscriber(tpose_topic, PoseStamped, self.target_pose_callback_)
		light_sensor=rospy.Subscriber("/{}/sensor_readings".format(robot_namespace), Float32MultiArray, self.light_callback_)
		
		rate=rospy.Rate(self.awake_freq)

		while (not rospy.is_shutdown()):
			if not(self.robot_pose==None or self.target_pose==None or self.light_readings==None):
				print(self.robot_pose)
				print('target:',self.target_pose)
				print('light:',self.light_readings)
				self.robot_loc_stack.append(pose2xz(self.robot_pose))
				self.target_loc_stack.append(pose2xz(self.target_pose))
				self.light_reading_stack.append(np.array(self.light_readings))
			rate.sleep()

		if save_data:
			np.savetxt('robot_loc_{}.txt'.format(robot_namespace),np.array(self.robot_loc_stack),delimiter=',')
			np.savetxt('target_loc_{}.txt'.format(target_namespace),np.array(self.target_loc_stack),delimiter=',')
			np.savetxt('light_readings_{}.txt'.format(robot_namespace),np.array(self.light_reading_stack),delimiter=',')

		print('Calculating Coefficients...')
		np.savetxt('coefs_{}.txt'.format(robot_namespace),
						cc(np.array(self.robot_loc_stack),
							np.array(self.target_loc_stack),
							np.array(self.light_reading_stack),fit_type=fit_type))
		
		
if __name__ == '__main__':
	arguments = len(sys.argv) - 1

	# print(arguments,sys.argv)
	position = 1
	# Get the robot name passed in by the user
	robot_namespace=''
	if arguments>=position:
		robot_namespace=sys.argv[position]

	position = 2
	# Get the target name passed in by the user
	target_namespace='Lamp'
	if arguments>=position:
		target_namespace=sys.argv[position]

	position = 3
	# Get the desired fitting mode: loss wrt dist or light readings
	fit_type="light_readings"
	if arguments>=position:
		fit_type=sys.argv[position]

	cmc=calibrate_meas_coef()
	cmc.record_and_calibrate(robot_namespace,target_namespace,save_data=True,fit_type=fit_type)
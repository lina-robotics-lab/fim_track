import rospy
from utils.RemotePCCodebase import *
from std_msgs.msg import Float32MultiArray
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
		self.robot_yaw_stack=[]
		self.light_reading_stack=[]
		self.rhats=[]

		self.C1=None
		self.C0=None
		self.k=None
		self.b=None

		rospy.Subscriber(self.rpose_topic, self.pose_type, self.robot_pose_callback_)
		rospy.Subscriber(self.light_topic, Float32MultiArray, self.light_callback_)
		rospy.Subscriber(self.coefs_topic, Float32MultiArray, self.sensor_coef_callback_)


	def sensor_coef_callback_(self,data):
		coefs=data.data
		self.C1,self.C0,self.k,self.b=coefs

	def robot_pose_callback_(self,data):
		self.robot_pose=data

	def light_callback_(self,data):
		self.light_readings=data.data


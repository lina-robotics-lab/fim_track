#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
import numpy as np
import sys

class spin_and_collect(object):
	"""The object that controls a turtlebot to spin w.r.t. its own axis and collect light sensor readings."""
	def __init__(self,awake_freq=10):
		self.awake_freq_=awake_freq
		self.pub_=None
		self.reading_records=[]
		self.node_init_=False
		self.spin_angular_vel_=1

	# Public methods
	# Each spin_and_collect object contains its one and only node.
	def init_node(self):
		if self.node_init_:
			print("The spin_and_collect node has already been initialized.")
			return
		# Initialize a ROS node that handles spin and collect.
		rospy.init_node('spin_and_collect',anonymous=True)
		self.node_init_=True

	def simple_collect(self,robot_namespace='',total_time=np.inf):
		if not self.node_init_:
			print("The spin_and_collect node is not initialized yet. Call self.init_node() first.")
			return

		r = rospy.Rate(self.awake_freq_)
		# Initialize the light sensor reading listener.
		self.collect_start_(robot_namespace)
		counter=0
		while (not rospy.is_shutdown()) and counter<total_time*self.awake_freq_:
				counter+=1
				# print(counter,total_time*self.awake_freq_)
				r.sleep()


	def spin_and_collect(self,robot_namespace='',total_time=0):# By default rotate for 0s.
		if not self.node_init_:
			print("The spin_and_collect node is not initialized yet. Call self.init_node() first.")
			return

		# Initialize the cmd_vel publisher and clock.
		r = rospy.Rate(self.awake_freq_)
		print('Spin the robot and collect readings')
		if robot_namespace!='':
			self.pub_=rospy.Publisher('/{}/cmd_vel'.format(robot_namespace),Twist,queue_size=10)
		else:
			self.pub_=rospy.Publisher('cmd_vel',Twist,queue_size=10)

		# Initialize the light sensor reading listener.
		self.collect_start_(robot_namespace)

		counter=0 # Keep spinning for total_time

		while (not rospy.is_shutdown()) and counter<total_time*self.awake_freq_:
				self.spin_()
				counter+=1
				# print(counter,total_time*self.awake_freq_)
				r.sleep()

		# Elegantly stop the robot.
		self.stop_()
		self.pub_=None
		
	# Private methods
	def callback_(self,data):
		print(np.array(data.data))
		self.reading_records.append(list(data.data))

	def collect_start_(self,robot_namespace=''):
		print('Initializing light_listener')
		if robot_namespace!='':
			rospy.Subscriber("/{}/sensor_readings".format(robot_namespace), Float32MultiArray, self.callback_)
		else:
			rospy.Subscriber("sensor_readings", Float32MultiArray, self.callback_)
		

	def spin_(self):
		
		twist = Twist()
		twist.linear.x = 0.0
		twist.linear.y = 0.0
		twist.linear.z = 0.0
		twist.angular.x = 0.0
		twist.angular.y = 0.0
		twist.angular.z = self.spin_angular_vel_
		self.pub_.publish(twist)
		# print(twist)
		
	def stop_(self):
		twist = Twist()
		twist.linear.x = 0.0
		twist.linear.y = 0.0
		twist.linear.z = 0.0
		twist.angular.x = 0.0
		twist.angular.y = 0.0
		twist.angular.z = 0
		self.pub_.publish(twist)


if __name__=='__main__':
	arguments = len(sys.argv) - 1

	# print(arguments,sys.argv)
	position = 1
	# Get the robot name passed in by the user
	robot_namespace=''
	if arguments>=position:
		robot_namespace=sys.argv[position]
	
	position=2
	# Get the total time of spinning
	total_time=0
	if arguments>=position:
		total_time=float(sys.argv[position]) # Remember to parse the argument string to float
	


	awake_freq=10

	sc=spin_and_collect(awake_freq)
	sc.init_node()	
	sc.spin_and_collect(robot_namespace,total_time)
	
	print('Max Reading:',np.max(sc.reading_records))
	print('Min Reading:',np.min(sc.reading_records))
	
	np.savetxt('light_readings_{}.txt'.format(robot_namespace),np.array(sc.reading_records),delimiter=',')
	# print('light_readings_{}.txt'.format(robot_namespace))

#!/usr/bin/env python
import rospy
# from std_msgs.msg import Twist
from geometry_msgs.msg import Twist
from sensor_reading_listener import light_listener
from std_msgs.msg import Float32MultiArray
import numpy as np


class spin_and_collect(object):
	"""docstring for spin_and_collect"""
	def __init__(self,pub):
		self.pub=pub
		self.storage=[]
		
	def callback(self,data):
		print(np.array(data.data))
		self.storage.append(list(data.data))

	def collect_start(self):
		print('Initializing light_listener')
		rospy.Subscriber("/turtlebot3/sensor_readings", Float32MultiArray, self.callback)
		

	def spin(self):
		
		twist = Twist()
		twist.linear.x = 0.0
		twist.linear.y = 0.0
		twist.linear.z = 0.0
		twist.angular.x = 0.0
		twist.angular.y = 0.0
		twist.angular.z = 2
		self.pub.publish(twist)
		# print(twist)
		
	def stop(self):
		twist = Twist()
		twist.linear.x = 0.0
		twist.linear.y = 0.0
		twist.linear.z = 0.0
		twist.angular.x = 0.0
		twist.angular.y = 0.0
		twist.angular.z = 0
		self.pub.publish(twist)


if __name__=='__main__':
	total_time=30
	awake_freq=10

	robot_name='/turtlebot3'
	rospy.init_node('spin_and_collect',anonymous=True)
	
	# First spin, then collect
	r = rospy.Rate(awake_freq)
	print('Spin the robot and collect readings')
	pub=rospy.Publisher('{}/cmd_vel'.format(robot_name),Twist,queue_size=10)
	
	sc=spin_and_collect(pub)

	
	sc.collect_start()

	counter=0 # Keep spinning for total_time

	while not rospy.is_shutdown() and counter<total_time*awake_freq:
			sc.spin()
			counter+=1
			r.sleep()

	# Elegantly stop the robot.
	sc.stop()
	print(np.array(sc.storage))
	np.savetxt('light_readings.txt',np.array(sc.storage),delimiter=',')
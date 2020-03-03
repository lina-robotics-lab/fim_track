#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray
import sys

def log_light(data):
	rospy.loginfo(data.data)

def light_listener(callback,robot_namespace=''):
	print('Initializing light_listener')
	rospy.init_node('light_listener',anonymous=True)
	if not robot_namespace=='':
		rospy.Subscriber("{}/sensor_readings".format(robot_namespace), Float32MultiArray, callback)
	else:
		rospy.Subscriber("sensor_readings", Float32MultiArray, callback)
	rospy.spin()

if __name__=='__main__':

	# Get the robot name passed in by the user
	arguments = len(sys.argv) - 1
	position = 1
	robot_namespace=''
	if arguments>=position:
		robot_namespace='/'+sys.argv[position]

	light_listener(log_light,robot_namespace)

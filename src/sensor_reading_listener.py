#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32MultiArray

def log_light(data):
	rospy.loginfo(data.data)

def light_listener(callback):
	print('Initializing light_listener')
	rospy.init_node('light_listener',anonymous=True)
	rospy.Subscriber("/turtlebot3/sensor_readings", Float32MultiArray, callback)
	rospy.spin()

if __name__=='__main__':
	light_listener(log_light)

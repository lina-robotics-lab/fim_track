#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
import numpy as np
import sys
from spin_and_collect import spin_and_collect

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
	total_time=np.inf
	if arguments>=position:
		total_time=float(sys.argv[position]) # Remember to parse the argument string to float
	


	awake_freq=10

	sc=spin_and_collect(awake_freq)
	sc.init_node()	
	sc.simple_collect(robot_namespace,total_time)
	
	print(np.array(sc.reading_records))
	
	np.savetxt('light_readings_{}.txt'.format(robot_namespace),np.array(sc.reading_records),delimiter=',')
	print('light_readings_{}.txt'.format(robot_namespace))
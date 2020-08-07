#!/usr/bin/env python3

from utils.RemotePCCodebase import prompt_pose_type_string,get_sensor_names,stop_twist

import argparse
import roslaunch
import rospy
import rosparam
import subprocess
import rospkg
import sys
import re
from geometry_msgs.msg import Twist



def launch_tracking_suite(pose_type_string,n_robots,local_track_alg,sensor_names):
	# Using create a launcher node.
	rospy.init_node('target_tracking_suite', anonymous=False)
	uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
	roslaunch.configure_logging(uuid)

	launch=roslaunch.scriptapi.ROSLaunch()

	launch.start()


	# Start launching nodes. First, the location estimation package.
	args=" ".join([pose_type_string])
	# print(args)
	node=roslaunch.core.Node(package='fim_track',node_type='location_estimation.py',name='location_estimation',namespace='/',args=args,output='screen')
	launch.launch(node)

	# Next, the multi_robot_controller for path planning.
	args=" ".join([pose_type_string])
	node=roslaunch.core.Node(package='fim_track',node_type='multi_robot_controller.py',name='multi_robot_controller',namespace='/',args=args,output='screen')
	launch.launch(node)

	# Next, as many single_robot_controller as specified
	for i in range(n_robots):
		args=" ".join([pose_type_string,sensor_names[i],local_track_alg])
		node=roslaunch.core.Node(package='fim_track',node_type='single_robot_controller.py',name='single_robot_controller_{}'.format(i),namespace='/',args=args,output='screen')
		launch.launch(node)
	# Start the logging utility
	args = " ".join([pose_type_string])
	node = roslaunch.core.Node(package='fim_track',node_type='tracking_log.py',name = 'tracking_log',namespace='/',args = args,output = 'screen')
	launch.launch(node)

	
	try:
		launch.spin()
	except rospy.exceptions.ROSInterruptException:
		pass
	finally:
		# After Ctrl+C is pressed.
		names = get_sensor_names()
		vel_pubs=[rospy.Publisher('/{}/cmd_vel'.format(name),Twist,queue_size=10) for name in names]
		print("Stopping the robots...")

		for i in range(15):
			if (not rospy.is_shutdown()):
				rate=rospy.Rate(10)
				for pub in vel_pubs:
					pub.publish(stop_twist())
				rate.sleep()
		print('Target tracking ends')




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--pose_type_string")
	args = parser.parse_args()
	
	if not args.pose_type_string:
		pose_type_string=prompt_pose_type_string()
	else:
		pose_type_string=args.pose_type_string
	
	sensor_names = get_sensor_names()

	n_robots = len(sensor_names)
	print("{} Mobile Sensors Detected".format(n_robots))
	print(sensor_names)
	print("pose type:",pose_type_string)

	local_track_algs = dict()
	local_track_algs['l']='LQR'
	local_track_algs['t']='TurnAndGo'

	resp=input('\n'.join(['Local Tracking Algorithm To Use','l=>LQR','t=>TurnAndGo:']))


	launch_tracking_suite(pose_type_string,n_robots,local_track_algs[resp],sensor_names)
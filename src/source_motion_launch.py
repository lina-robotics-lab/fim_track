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
	rospy.init_node('source_movement', anonymous=False)
	uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
	roslaunch.configure_logging(uuid)

	launch=roslaunch.scriptapi.ROSLaunch()

	launch.start()


	# Start launching nodes. First, the location estimation package.
	args=" ".join([pose_type_string])
	# print(args)
	
	# Next, the multi_robot_controller for path planning.
	args=" ".join([pose_type_string])
	node=roslaunch.core.Node(package='fim_track',node_type='multi_source_controller.py',name='multi_source_controller',namespace='/',args=args,output='screen')
	launch.launch(node)

	# Next, as many single_robot_controller as specified
	for i in range(n_robots):
		args=" ".join([pose_type_string,sensor_names[i],local_track_alg])
		node=roslaunch.core.Node(package='fim_track',node_type='single_robot_controller.py',name='single_robot_controller_{}'.format(i),namespace='/',args=args,output='screen')
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
		print('Test waypoint tracking ends')




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--pose_type_string")
	args = parser.parse_args()
	
	if not args.pose_type_string:
		pose_type_string=prompt_pose_type_string()
	else:
		pose_type_string=args.pose_type_string
	

	source_names = ['target_0']

	launch_tracking_suite(pose_type_string,len(source_names),"LQR",source_names)
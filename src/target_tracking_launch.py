#!/usr/bin/env python3

from RemotePCCodebase import prompt_pose_type_string

import roslaunch
import rospy
import rosparam
import subprocess
import rospkg
import sys


def launch_tracking_suite(pose_type_string,n_robots):
	# Using create a launcher node.
	rospy.init_node('target_tracking_suite', anonymous=False)
	uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
	roslaunch.configure_logging(uuid)

	launch=roslaunch.scriptapi.ROSLaunch()

	launch.start()

	# Start launching nodes. First, the location estimation package.
	args=" ".join([pose_type_string,str(n_robots)])
	node=roslaunch.core.Node(package='fim_track',node_type='location_estimation.py',name='location_estimation',namespace='/',args=args,output='screen')
	launch.launch(node)

	# Next, the multi_robot_controller for path planning.
	args=" ".join([pose_type_string,str(n_robots)])
	node=roslaunch.core.Node(package='fim_track',node_type='multi_robot_controller.py',name='multi_robot_controller',namespace='/',args=args,output='screen')
	launch.launch(node)

	# Finally, as many single_robot_controller as specified
	for i in range(n_robots):
		args=" ".join([pose_type_string,str(i)])
		node=roslaunch.core.Node(package='fim_track',node_type='single_robot_controller.py',name='single_robot_controller_{}'.format(i),namespace='/',args=args,output='screen')
		launch.launch(node)

	try:
		launch.spin()
	except rospy.exceptions.ROSInterruptException:
		# After Ctrl+C is pressed.
		pass




if __name__ == '__main__':
	arguments=len(sys.argv)-1
	if arguments<=0:
		pose_type_string=prompt_pose_type_string()
		n_robots=int(input('The number of mobile sensors:'))
	else:
		if arguments>=1:
			pose_type_string=sys.argv[1]
		if arguments>=2:
			n_robots=int(sys.argv[2])

	launch_tracking_suite(pose_type_string,n_robots)
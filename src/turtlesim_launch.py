#!/usr/bin/env python3
import roslaunch
import rospy
import rosparam
import subprocess
import rospkg
import os
from turtlesim.srv import Spawn,Kill
import numpy as np




def launch_simulation(sensor_poses=[],target_poses=[]):
	'''
		Simplified pose format: [x,y,z,Yaw]. 

		The poses are lists of 4-vectors.

		The number of sensors and targets to use is automatically determined by the dimensions of poses passed in.
	'''

	# Using roslaunch api to launch an empty world.
	rospy.init_node('fim_track_simulation', anonymous=False)
	uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
	roslaunch.configure_logging(uuid)

	launch=roslaunch.scriptapi.ROSLaunch()


	# Remark: now we are reusing the empty world launch file provided by gazebo_ros to provide a ground plane. More complicated worlds can be used in the future.
	launch.start()

	# Launch the turtlesim enviroment.
	turtlesim_namespace="/"
	node=roslaunch.core.Node(package='turtlesim',node_type='turtlesim_node',name='turtlesim_environment',namespace=turtlesim_namespace,output='screen')
	launch.launch(node)


	# Start spawning objects into the world.
	rospy.wait_for_service('{}/spawn'.format(turtlesim_namespace))
	rospy.wait_for_service('{}/kill'.format(turtlesim_namespace))
	spawn=rospy.ServiceProxy('{}/spawn'.format(turtlesim_namespace),Spawn)
	kill=rospy.ServiceProxy('{}/kill'.format(turtlesim_namespace),Kill)

	
	# Remove the default turtle from the world
	kill('turtle1'.format(turtlesim_namespace))


	# Specify target names and mobile sensor names.
	target_names=['target_{}'.format(i) for i in range(len(target_poses))]
	mobile_sensor_names=['mobile_sensor_{}'.format(i) for i in range(len(sensor_poses))]

	
	# Spawn the targets, start emitting virtual lights
	
	for i, pose in enumerate(target_poses):
		try:
			resp=spawn(pose[0],pose[1],pose[2],target_names[i])
	
			virtual_light_node=roslaunch.core.Node(package='fim_track',node_type='virtual_light.py',name='virtual_light_for_{}'.format(target_names[i]),namespace='',args=target_names[i],output='screen')
	
			launch.launch(virtual_light_node)
	
		except rospy.exceptions.ROSException:
			print('Object Spawning Error')
			pass

	# Spawn the mobile sensors, start receiving virtual lights.
	
	for i, pose in enumerate(sensor_poses):
		try:
	
			resp=spawn(pose[0],pose[1],pose[2],mobile_sensor_names[i])
			arg_string="turtlesimPose"+" "+mobile_sensor_names[i]+" "+" ".join(target_names)

			virtual_sensor_node=roslaunch.core.Node(package='fim_track',node_type='virtual_sensor.py',name='virtual_sensor_for_{}'.format(mobile_sensor_names[i]),namespace='',args=arg_string,output='screen')
	
			launch.launch(virtual_sensor_node)
	
		except rospy.exceptions.ROSException:
			print('Object Spawning Error')
			pass


	try:
		launch.spin()
	except rospy.exceptions.ROSInterruptException:
		# After Ctrl+C is pressed.
		pass

if __name__ == '__main__':

	# Specify the initial sensor poses and target poses here.
	# Simplified pose format: [x,y,theta]. 
	# The number of sensors and targets to use is automatically determined by the dimensions of poses passed in.
	

	############ Setup for source seeking experiment
	target_loc = [6.0,6.0]
	target_poses=[[target_loc[0],target_loc[1],3.14/2]]
	
	sensor_poses=[
	[2,2,-3.04],\
	[1,2,-3.04],\
	[3,2,-3.04]]
	

	############ Setup for Location Estimation Experiment

	# target_loc = np.array([1.2,1.2])
	# radius = 1
	
	# sensor_angles = [0,-np.pi/3,np.pi/3]
	# sensor_angles = [0,np.pi*2/3,np.pi*4/3]
	# sensor_angles = [0,-np.pi/6,np.pi/6]

	# sensor_locs = target_loc + radius*np.array([np.cos(sensor_angles), np.sin(sensor_angles)]).T

	# target_poses=[[target_loc[0],target_loc[1],0]]

	# sensor_poses=np.vstack([sensor_locs.T,np.zeros(3)]).T


	noise_level=input('Additive Gaussian Noise Std(a non-negative number):')
	rospy.set_param('noise_level',float(noise_level))
	# Specify the path to a basis launch file. It usually contains information about the .world file.
	# Here we use the empty world launch file provided by gazebo_ros.
	launch_simulation(sensor_poses=sensor_poses,target_poses=target_poses)




	
#!/usr/bin/env python3
import roslaunch
import rospy
import rosparam
import subprocess
import rospkg
import os
from turtlesim.srv import Spawn,Kill

def spawn_object(model,new_object_name,launch,x,y):
	'''
		model: a string. Should be either motion_lamp or turtlebot3_burger, or anything that exists in the models directory of fim_track.
		new_object_name: a string.
		launch: the launch object to be passed in.
	'''
	
	pass



def launch_simulation(sensor_poses=[],target_poses=[]):
	'''
		Simplified pose format: [x,y,z,Yaw]. 

		The poses are lists of 4-vectors.

		The number of sensors and targets to use is automatically determined by the dimensions of poses passed in.
	'''

	# Using roslaunch api to launch an empty world.
	rospy.init_node('fim_track_simulation', anonymous=True)
	uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
	roslaunch.configure_logging(uuid)

	launch=roslaunch.scriptapi.ROSLaunch()


	# Remark: now we are reusing the empty world launch file provided by gazebo_ros to provide a ground plane. More complicated worlds can be used in the future.
	launch.start()

	# Launch the turtlesim enviroment.
	turtlesim_namespace="/turtlesim"
	node=roslaunch.core.Node(package='turtlesim',node_type='turtlesim_node',namespace=turtlesim_namespace,output='screen')
	launch.launch(node)


	# Start spawning objects into the world.
	rospy.wait_for_service('{}/spawn'.format(turtlesim_namespace))
	rospy.wait_for_service('{}/kill'.format(turtlesim_namespace))
	spawn=rospy.ServiceProxy('{}/spawn'.format(turtlesim_namespace),Spawn)
	kill=rospy.ServiceProxy('{}/kill'.format(turtlesim_namespace),Kill)

	
	# Remove the default turtle from the world
	kill('turtle1'.format(turtlesim_namespace))

	# Spawn the mobile sensors
	mobile_sensor_names=['mobile_sensor_{}'.format(i) for i in range(len(sensor_poses))]
	for i, pose in enumerate(sensor_poses):
		try:
			resp=spawn(pose[0],pose[1],pose[2],mobile_sensor_names[i])
		except rospy.exceptions.ROSException:
			print('Object Spawning Error')
			pass

	# Spawn the targets
	target_names=['target_{}'.format(i) for i in range(len(target_poses))]
	for i, pose in enumerate(target_poses):
		try:
			resp=spawn(pose[0],pose[1],pose[2],target_names[i])
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
	sensor_poses=[[2,3,0],\
	[2,2,0],\
	[2,4,0]]
	target_poses=[[8,8,0]]

	# Specify the path to a basis launch file. It usually contains information about the .world file.
	# Here we use the empty world launch file provided by gazebo_ros.
	launch_simulation(sensor_poses=sensor_poses,target_poses=target_poses)




	
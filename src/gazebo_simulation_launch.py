#!/usr/bin/env python3
import roslaunch
import rospy
import rosparam
import subprocess
import rospkg
import os

def spawn_object(model,new_object_name,launch,x,y,z,Y):
	'''
		model: a string. Should be either motion_lamp or turtlebot3_burger, or anything that exists in the models directory of fim_track.
		new_object_name: a string.
		launch: the launch object to be passed in.
	'''


	# Retrieve the model description parameter from rosparam server
	fim_dir=rospkg.RosPack().get_path('fim_track')

	stream = os.popen('xacro'+' --inorder '+fim_dir+"/models/"+'{}.urdf.xacro'.format(model))
	model_description = stream.read()

	# Set the param new_object_name/robot_description in rosparam. Useful in subsequent node initializations.
	rospy.set_param(new_object_name,{"robot_description":model_description})

	# Launch the robot_state_publisher node. This publisher makes the object observable and controllable via ROS after being brought to life in Gazebo simulator.
	node=roslaunch.core.Node(package='robot_state_publisher',node_type='robot_state_publisher',namespace=new_object_name,output='screen')
	launch.launch(node)

	# Launch the spawn_model node. This bring the object to life in Gazebo simulator.
	node = roslaunch.core.Node(package='gazebo_ros', node_type='spawn_model',namespace=new_object_name,args='-urdf -model {} -x {} -y {} -z {} -Y {} -param robot_description'.format(new_object_name,x,y,z,Y))
	launch.launch(node)


def launch_simulation(sensor_poses=[],target_poses=[],basis_launch_file=None):
	'''
		Simplified pose format: [x,y,z,Yaw]. 

		The poses are lists of 4-vectors.

		The number of sensors and targets to use is automatically determined by the dimensions of poses passed in.

		If basis_launch file is not provided, the empty world of gazebo_ros will be launched.
	'''

	# Using roslaunch api to launch an empty world.
	rospy.init_node('fim_track_simulation', anonymous=True)
	uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
	roslaunch.configure_logging(uuid)

	launch=roslaunch.scriptapi.ROSLaunch()

	if basis_launch_file==None:
		gazebo_ros_dir=rospkg.RosPack().get_path('gazebo_ros')
		basis_launch_file='{}/launch/empty_world.launch'.format(gazebo_ros_dir)

	# Remark: now we are reusing the empty world launch file provided by gazebo_ros to provide a ground plane. More complicated worlds can be used in the future.
	launch.parent = roslaunch.parent.ROSLaunchParent(uuid, [basis_launch_file])
	launch.start()


	# Spawn the mobile sensors into the empty world
	for i, pose in enumerate(sensor_poses):
		spawn_object("turtlebot3_burger",'mobile_sensor_{}'.format(i),launch,x=pose[0],y=pose[1],z=pose[2],Y=pose[3])

	# Spawn the targets into the empty world
	for i, pose in enumerate(target_poses):
		spawn_object("motion_lamp",'target_{}'.format(i),launch,x=pose[0],y=pose[1],z=pose[2],Y=pose[3])

	try:
		launch.spin()
	finally:
	  # After Ctrl+C, stop all nodes from running
		pass

if __name__ == '__main__':

	# Specify the initial sensor poses and target poses here.
	# Simplified pose format: [x,y,z,Yaw]. 
	# The number of sensors and targets to use is automatically determined by the dimensions of poses passed in.
	sensor_poses=[[4,0,0,0],\
	[4,0.5,0,0],\
	[4,-0.5,0,0]]
	target_poses=[[0,0,0,0]]

	# Specify the path to a basis launch file. It usually contains information about the .world file.
	# Here we use the empty world launch file provided by gazebo_ros.
	gazebo_ros_dir=rospkg.RosPack().get_path('gazebo_ros')
	empty_world='{}/launch/empty_world.launch'.format(gazebo_ros_dir)

	launch_simulation(sensor_poses=sensor_poses,target_poses=target_poses,basis_launch_file=empty_world)




	
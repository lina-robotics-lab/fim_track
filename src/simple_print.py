#!/usr/bin/env python3
import roslaunch
import rospy
import rosparam
import subprocess
import rospkg
import os

def spawn_objects(model,n_objects,launch,x,y,z,Y):

	for i in range(n_objects):

		new_object_name = "{}_{}".format(model,i)

		# Retrieve the model description parameter from rosparam server

		stream = os.popen('xacro'+' --inorder '+fim_dir+"/models/"+'{}.urdf.xacro'.format(model))
		model_description = stream.read()
		# model_description=os.system()

		# Set the param new_object_name/robot_description in rosparam. Useful in subsequent node initializations.
		rospy.set_param(new_object_name,{"robot_description":model_description})

		node=roslaunch.core.Node(package='robot_state_publisher',node_type='robot_state_publisher',namespace=new_object_name,output='screen')
		launch.launch(node)

		node = roslaunch.core.Node(package='gazebo_ros', node_type='spawn_model',namespace=new_object_name,args='-urdf -model {} -x {} -y {} -z {} -Y {} -param robot_description'.format(new_object_name,x,y,z,Y))
		launch.launch(node)





rospy.init_node('fim_track_simulation', anonymous=True)
uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
roslaunch.configure_logging(uuid)

fim_dir=rospkg.RosPack().get_path('fim_track')

launch=roslaunch.scriptapi.ROSLaunch()
launch.parent=roslaunch.parent.ROSLaunchParent(uuid, ["{}/launch/fim_track_gazebo_skeleton.launch".format(fim_dir)])
launch.start()



n_sensor=1

spawn_objects("turtlebot3_burger",n_sensor,launch,0,0,0,0)

n_target=1

spawn_objects("motion_lamp",n_target,launch,1,0,0,0)


# rosparam.set_param(new_object_name,'{"robot_state_publisher/":}')

try:
	launch.spin()
finally:
  # After Ctrl+C, stop all nodes from running
	launch.shutdown()
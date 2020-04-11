#!/usr/bin/env python2

import rospy
import rospkg


from gazebo_msgs.srv import SpawnModel,DeleteModel
from geometry_msgs.msg import *

if __name__ == '__main__':
    print("Waiting for gazebo services...")
    rospy.init_node("spawn_products_in_bins",anonymous=True)
    rospy.wait_for_service("gazebo/spawn_urdf_model")
    rospy.wait_for_service("gazebo/delete_model")
    print("Got it.")

    spawn_model = rospy.ServiceProxy("gazebo/spawn_urdf_model", SpawnModel)
    delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)	
	
    tb3_dir=rospkg.RosPack().get_path('turtlebot3_description')

    with open("{}/urdf/turtlebot3_burger.urdf.xacro".format(tb3_dir), "r") as f:
        product_xml = f.read()

    
    for num in range(12):
        item_name = "product_{0}_0".format(num)
        
    for num in range(12):
        bin_y   =   2.8 *   (num    /   6)  -   1.4 
        bin_x   =   0.5 *   (num    %   6)  -   1.5
        item_name   =   "product_{0}_0".format(num)
        print("Spawning model:{}".format(item_name))
        item_pose   =   Pose(position=Point(x=bin_x, y=bin_y,    z=2))
	delete_model(item_name)	
        spawn_model(item_name, product_xml, item_name, item_pose, "world")

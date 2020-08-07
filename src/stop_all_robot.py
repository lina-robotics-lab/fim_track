#!/usr/bin/env python3
import rospy

from utils.RemotePCCodebase import stop_twist,get_sensor_names

from geometry_msgs.msg import Twist

		

if __name__ == '__main__':
	rospy.init_node('stop_all_robot')
	names=get_sensor_names()
	vel_pubs=[rospy.Publisher('/{}/cmd_vel'.format(name),Twist,queue_size=10) for name in names]
	while(not rospy.is_shutdown()):
		for pub in vel_pubs:
			pub.publish(stop_twist())
#!/usr/bin/env python3

'''
This script recording the the trajectories of all moving objects in the world of ROS
'''
import rospy,rostopic
import re
from math import atan2
from geometry_msgs.msg import PoseStamped,Pose, Twist
from nav_msgs.msg import Odometry
from collections import deque

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib import style
import json
import numpy as np




def quaternion2yaw(q):
	siny_cosp = 2 * (q.w * q.z + q.x * q.y)
	cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
	return atan2(siny_cosp, cosy_cosp)
def yaw_from_odom(odom):
	return quaternion2yaw(odom.pose.pose.orientation)

def xy_from_odom(odom):
	return [odom.pose.pose.position.x,odom.pose.pose.position.y]

def namespace_from_topic(topic):
	ns=re.search("/.*/",topic) # Search for the first segment enclosed by two forward slashes.
	return ns.group()[:-1] # Return the namespace excluding the final forward slash.

def animate(i,trajectories):
	pass

class OdomPathMonitor(object):
	"""An OdomPathMonitor only keeps the odom data of one object"""
	def __init__(self, ros_namespace,maxlen=300):
		super(OdomPathMonitor, self).__init__()
		self.namespace=ros_namespace
		self.odom_listener = rospy.Subscriber('{}/odom'.format(ros_namespace),Odometry,self._callback) 
		self.current_odom=Odometry()
		self.odoms=deque(maxlen=maxlen)
	
	def _callback(self,odom):
		self.current_odom=odom
	def update_trajectory(self):
		self.odoms.append(self.current_odom)
	
	def positions(self):
		return [xy_from_odom(o) for o in self.odoms]
	
	def angles(self):
		return [yaw_from_odom(o) for o in self.odoms]

def main():
	rospy.init_node('monitor_robot_path',anonymous=True)

	odom_topics=[]
	all_published_topics=rospy.get_published_topics()
	
	for topic in all_published_topics:
		if topic[1]=='nav_msgs/Odometry': # Odometry is typically used by our manually spawned objects.
			odom_topics.append(topic[0])

	monitors=[]
	if len(odom_topics)>0:
		for topic in odom_topics:
			ns=namespace_from_topic(topic)
			monitors.append(OdomPathMonitor(ns))

	r=rospy.Rate(50)
	trajectories=[]
	
	
	style.use('seaborn')

	fig=plt.figure()
	ax1=fig.add_subplot(1,1,1)

	def animation(i):
		for m in monitors:
			m.update_trajectory()
			# angles=m.angles()
			# if len(angles)>0:
			# 	print(m.namespace,angles[-1])
		trajectories=dict({m.namespace:m.positions() for m in monitors})
			
		ax1.clear()
		for namespace,value in trajectories.items():
			value=np.array(value)
			ax1.plot(value[:,0],value[:,1],label=namespace,linestyle='-',marker='*', markersize='5')
		ax1.set_xlim((-10,10))
		ax1.set_ylim((-10,10))
		ax1.legend()

		r.sleep()
		
	ani=FuncAnimation(fig,animation,interval=10)
	plt.show()
if __name__ == '__main__':
	main()
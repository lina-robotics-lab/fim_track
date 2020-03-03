#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped,Pose, Twist
from std_msgs.msg import Float32MultiArray
import numpy as np
import sys
from RemotePCCodebase import multi_lateration,top_n_mean

## Multi-lateration Localization Algorithm. The shape of readings is (t*num_sensors,), the shape of sensor locs is (t*num_sensors,2). 
## For the algorithm to work, sensor_locs shall be not repetitive, and t*num_sensors shall be >=3.
def multi_lateration(readings,sensor_locs,C1=0.07,C0=1.29,k=15.78,b=-2.16):
    rhat=((readings-C0)/k)**(1/b)+C1
    A=2*(sensor_locs[-1,:]-sensor_locs)[:-1]
    B=rhat[:-1]**2-rhat[-1]**2+np.sum(sensor_locs[-1,:]**2)-np.sum(sensor_locs[:-1,:]**2,axis=1)
    qhat=np.linalg.pinv(A).dot(B)
    return qhat
class robot_listener:
	def __init__(self,robot_name):
		self.robot_name=robot_name
		self.rpose_topic="/vrpn_client_node/{}/pose".format(robot_name)
		self.light_topic="/{}/sensor_readings".format(robot_name)
		self.robot_pose=None
		self.light_readings=None

	def robot_pose_callback_(self,data):
		# print(data.pose)
		self.robot_pose=data.pose

	def light_callback_(self,data):
		# print(data.data)
		self.light_readings=data.data

class location_estimation:
	def __init__(self,robot_names,awake_freq=100):
		self.n_robots=len(robot_names)
		self.robot_names=robot_names
		self.listeners=[robot_listener(r) for r in robot_names]

		self.robot_loc_stack=[]
		self.light_reading_stack=[]

		self.awake_freq=awake_freq
		
		
	
	def pose2xz(self,pose):
		return np.array([pose.position.x,pose.position.z])
	
	def localize_target(self,look_back=100):
		scalar_readings=top_n_mean(self.light_reading_stack[-look_back:],2)
		sensor_locs=np.array(self.robot_loc_stack[-look_back:])
		return multi_lateration(scalar_readings,sensor_locs)

	def record_data(self):
		rospy.init_node('location_estimation',anonymous=True)
		for l in self.listeners:
			rospy.Subscriber(l.rpose_topic, PoseStamped, l.robot_pose_callback_)
			rospy.Subscriber(l.light_topic, Float32MultiArray, l.light_callback_)
		
		rate=rospy.Rate(self.awake_freq)

		while (not rospy.is_shutdown()):
			# Gather the latest readings from listeners
			for l in self.listeners:
				if not(l.robot_pose==None or l.light_readings==None):
				
					
					# print('name:',l.robot_name,'pose:',l.robot_pose,'light:',l.light_readings)
					self.robot_loc_stack.append(self.pose2xz(l.robot_pose))
					self.light_reading_stack.append(np.array(l.light_readings))
					'''
					The Magic to here: real-time localization algorithm.
					'''
					print(self.localize_target())
			rate.sleep()
		
		
		
if __name__=='__main__':
	arguments = len(sys.argv) - 1

	robot_names=[]
	for i in range(1,arguments+1,1):
		robot_names.append(sys.argv[i])

	target_name='Lamp'
	
	le=location_estimation(robot_names)
	le.record_data()


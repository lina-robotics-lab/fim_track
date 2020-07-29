#!/usr/bin/env python3
import sys
import rospy
from functools import partial
import numpy as np
import pickle as pkl

from std_msgs.msg import Float32MultiArray

from RemotePCCodebase import get_sensor_names,get_target_names,prompt_pose_type_string,toxy,timestamp
from robot_listener import robot_listener

class logger(object):
	"""docstring for logger"""
	def __init__(self, sensor_names,target_names,pose_type_string,awake_freq = 2):
		super(logger, self).__init__()
		
		self.awake_freq = awake_freq

		## Data containers
		self.est_loc_tag=['multi_lateration','intersection','ekf','pf']
		
		self.curr_est_locs=dict({tag:np.array([np.nan,np.nan]) for tag in self.est_loc_tag})
		self.est_locs_log = dict()

		self.target_locs = dict({r:[] for r in target_names})
		self.sensor_locs = dict({r:[] for r in sensor_names})

		self.curr_waypoints = dict()
		self.waypoint_log = dict({r:[] for r in sensor_names})


		## ROS setups
		rospy.init_node('tracking_log',anonymous=False)


		# Pose subscribers
		self.listeners=[robot_listener(r,pose_type_string) for r in sensor_names]
		self.target_listeners = [robot_listener(r,pose_type_string) for r in target_names]


		# Location subscribers
		self.est_loc_sub=dict()
		
		for tag in self.est_loc_tag:
			self.est_loc_sub[tag]=rospy.Subscriber('/location_estimation/{}'.format(tag),\
												Float32MultiArray, partial(self.est_loc_callback_,tag=tag))

		# Waypoint subscribers

		self.waypoint_sub=dict()
		for r in sensor_names:
			rospy.Subscriber('/{}/waypoints'.format(r),Float32MultiArray,\
							partial(self.waypoint_callback_,sensor_name=r))
		
	def est_loc_callback_(self,data,tag):
		self.curr_est_locs[tag]=np.array(data.data)
	
	def waypoint_callback_(self,data,sensor_name):
		self.curr_waypoints[sensor_name] = np.array(data.data).reshape(-1,2)


	def save_data(self,filepath):
		logs = dict()
		
		# A helper function
		def stack_items(locs_log):
			log = dict()
			for key,val in locs_log.items():
				if not val is None:
					if len(val)>0:
						log[key] = np.vstack(val)
			return log

		logs['est_locs_log'] = stack_items(self.est_locs_log)
		logs['sensor_locs'] = stack_items(self.sensor_locs)
		logs['target_locs'] = stack_items(self.target_locs)

		logs['waypoints'] = self.waypoint_log

		with open(filepath,'wb') as file:
			pkl.dump(logs,file)
		
	def start(self,filepath):
		rate=rospy.Rate(self.awake_freq)
		sim_time = 0
		while (not rospy.is_shutdown()):
			
			# Log estimated loc data
			for key,val in self.curr_est_locs.items():
				if not key in self.est_locs_log.keys():
					self.est_locs_log[key] = []
				if not val is None:
					# val = np.nan * np.ones(2,)	
					self.est_locs_log[key].append(val)

			# Log sensor and target loc data
			for l in self.listeners:
				if not l.robot_pose is None:
					self.sensor_locs[l.robot_name].append(toxy(l.robot_pose))
			for l in self.target_listeners:
				if not l.robot_pose is None:
					self.target_locs[l.robot_name].append(toxy(l.robot_pose))

			# Log waypoints
			for key,val in self.curr_waypoints.items():
				if not val is None:
					self.waypoint_log[key].append(val)

			self.save_data(filepath)
			rate.sleep()
		
		# After Ctrl+C is pressed, save the log data to pickle file.
		
		self.save_data(filepath)
		print('saving data at {}...'.format(filepath))
		


def main(argv):
	arguments=len(argv)-1
	if arguments<=0:
		pose_type_string=prompt_pose_type_string()
		# pose_type_string=""
	else:
		if arguments>=1:
			pose_type_string=sys.argv[1]
	
	sensor_names = get_sensor_names()
	target_names = get_target_names()
	log = logger(sensor_names,target_names,pose_type_string)

	filename = 'track_log_data'
	filepath = "/home/tianpeng/{}.pkl".format(filename)
		
	log.start(filepath=filepath)

if __name__ == '__main__':
	main(sys.argv)
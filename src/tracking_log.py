import numpy as np
import pickle as pkl

class logger(object):
	"""Data logger for the experiment"""
	def __init__(self, sensor_names,target_names,virtual_leader_names=[]):
		super(logger, self).__init__()
		## Data containers
		self.est_loc_tag=['multi_lateration','intersection','ekf','pf']
		
		self.est_locs_log =dict({tag:[] for tag in self.est_loc_tag})

		self.src_locs = dict({r:[] for r in target_names})
		self.sensor_locs = dict({r:[] for r in sensor_names})
		self.virtual_leader_locs = dict({r:[] for r in virtual_leader_names})


	def export(self):
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
		logs['src_locs'] = stack_items(self.src_locs)
		logs['virtual_leader_locs']=stack_items(self.virtual_leader_locs)
		# logs['scalar_readings'] = stack_items(self.scalar_readings_log)

		return logs 
	
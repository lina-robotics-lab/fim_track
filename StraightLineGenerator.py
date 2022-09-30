from utils.StraightLinePathPlanning import straight_line_path_planning
class StraightLineGenerator(object):
	"""docstring for StraightLineGenerator"""
	def __init__(self,pos_to_go,n_timesteps = 20):
		super(StraightLineGenerator, self).__init__()
		self.pos_to_go  = pos_to_go
		self.n_timesteps = n_timesteps

	def get_waypoints(self,qs):
		return straight_line_path_planning(self.pos_to_go,qs,len(qs),self.n_timesteps)

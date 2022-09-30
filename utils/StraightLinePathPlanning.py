import numpy as np

def straight_line_path_planning(q,ps,n_p,n_timesteps):
	"""
		q: Current location of the target.
		ps: Current locations of the mobile sensors.
		n_p: The number of sensors
		n_timesteps: The number of timesteps to plan ahead. 
		----------------------------------------------------------------------------------------
		Output: waypoints for each mobile sensor, Shape= (num_time_steps,num_sensors,2)
	"""


	p_trajs=[]

	q=q.reshape(-1,2)
	ps=ps.reshape(-1,2)

	direction = -(ps-q)
	direction = (direction.T/n_timesteps).T

	for i in range(n_timesteps):
		ps= ps + direction
		p_trajs.append(ps)

	return np.array(p_trajs).reshape(-1,n_p,2) # Shape= (num_time_steps,num_sensors,2)


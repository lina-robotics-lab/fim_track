import numpy as np
"""
Path Planning for concentric, equi-angular movement.
"""
def concentric_path_planning(R,ps,n_p,n_steps,max_linear_speed,dt):
	"""
		It suffices to specify only the starting and ending waypoints. 
		The spline fitting in motion generation will take care of the rest. 
	"""
	step_size = max_linear_speed*dt
	ps=ps.reshape(-1,2)

	phi = 2*np.pi/n_p

	CoM = np.mean(ps,axis = 0)

	destinations = CoM + R * np.array([[np.cos(phi*j),np.sin(phi*j)] for j in range(n_p)])

	reached = np.any(np.linalg.norm(ps-CoM,axis=1)>=R)

	dists = np.repeat(step_size* np.arange(n_steps+1),n_p).reshape(-1,n_p)
	
	directions = np.expand_dims((((destinations-CoM).T/np.linalg.norm(destinations-ps,axis=1)).T),axis=2)

	directions = np.transpose(directions,(2,0,1))

	directions = np.repeat(directions,len(dists),axis=0)

	waypoints = directions*dists[:,:,np.newaxis] + CoM
	return waypoints,reached # shape = (2,n_p,space_dim=2)
    
    

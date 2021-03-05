from jax import jit, jacfwd
import jax.numpy as jnp
import numpy as np
from utils.regions import CircleInterior,Rect2D
"""
	Movement direction calculation follows a simplified version of equation (10) in
	
	Moore, B. J., & Canudas-de-Wit, C. (2010). Source seeking via collaborative measurements by a circular formation of agents. Proceedings of the 2010 American Control Conference, 6417–6422. https://doi.org/10.1109/ACC.2010.5531473
   
	Different from Moore et al., we achieve a equi-angular circular formation by doing projected gradient ascent of trace(separation matrix) on a circle.
	
	We then move the center of the circle using the formula in equation (10) prescribed by Moore et al.
""" 
def sep_func(ps):
	
	CoM = jnp.mean(ps,axis=0)
	A = ps-CoM
	return jnp.linalg.det(A.T.dot(A))

def mutual_separation_path_planning(R,ps,n_p,n_timesteps,\
	max_linear_speed,dt,scalar_readings,xlim = (0,2.4),ylim = (0,4.5),CoM_motion_gain=2,\
	f_dLdp=None):
	step_size = max_linear_speed*dt
	ps=ps.reshape(-1,2)

	if f_dLdp is None:
		f_dLdp = jit(jacfwd(sep_func))
	p_trajs=[]
	reached=False
	for i in range(n_timesteps):
			# Calculate the gradient
			grad=f_dLdp(ps)

			if np.any(np.isnan(grad)): # This is a hack that gets rid of degenerate gradient with random directions
				grad = np.random.random(grad.shape)
			
			grad=grad.reshape(-1,2)
			grad_sizes=np.linalg.norm(grad,axis=1)
			grad_sizes[grad_sizes==0]=1 # Handle the case where the partial derivative is zero.

			update_steps=(grad.T/np.max(grad_sizes) * step_size).T # Calculate the update steps to be applied to ps

			candid_ps=np.array(ps+update_steps) # Calculate the direct update 


			
			# Slightly move the CoM towards the direction where signal is the strongest.
			weights = np.abs(scalar_readings)/np.sum(np.abs(scalar_readings)) # Create a vector of convex combination weights.

			CoM = np.mean(ps,axis=0)
			weighted_CoM = np.sum(ps*weights[:,np.newaxis],axis=0)
			diff = weighted_CoM-CoM
		
			# CoM_vel = CoM_motion_gain * np.mean(np.linalg.norm(ps-CoM,axis=1))
			CoM_vel = CoM_motion_gain * max_linear_speed
			CoM += diff/np.linalg.norm(diff) * CoM_vel * dt
			
			# Prevent the ps from getting too far away by doing projection
			for j in range(len(candid_ps)):
				candid_ps[j,:] = Rect2D(xlim,ylim).project_point(\
					CircleInterior(CoM,R).project_point(candid_ps[j,:])\
					)
			
			p_trajs.append(candid_ps)
			
			if np.max(np.linalg.norm(ps-candid_ps,axis=1))<0.1 and i == 0:
				reached = True

				# break
			ps=candid_ps # Update ps.
			
	p_trajs = np.array(p_trajs)
	return p_trajs
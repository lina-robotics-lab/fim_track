import numpy as np
def polar_projection(q,ps,r):    
    """ 
        Return the projection of a set of points ps onto a circle centered at q with radius r
    """
    q=q.reshape(-1,2)
    ps=ps.reshape(-1,2)
    dists=np.linalg.norm(ps-q,axis=1)

    dists[dists==0]=1 # Handling the zero distances
    ps_proj=((ps-q).T/dists * r).T +q
    return ps_proj

def FIM_ascent_path_planning(f_dLdp,q,ps,n_p,n_timesteps,max_linear_speed,dt,epsilon,region=None):
    """
        f_dLdp: a function handle, f_dLdp(q,ps)=dLdp.
        q: Current location of the target.
        ps: Current locations of the mobile sensors.
        n_p: The number of sensors
        n_timesteps: The number of timesteps to plan ahead. 
        
            The total time horizon T will be T=n_timesteps*dt.
        
        max_linear_speed: the linear speed limit to be set on the mobile sensors.
        dt: the time differences between two consecutive waypoints.
        
            The update step size will be a constant = max_linear_speed * dt
        
        epsilon: when the planned trajectories end, how far away should they be to the target.
        region: If None, unconstraint gradient ascent is performed. Otherwise, region should be
        of class Region as defined in regions.py file in fim_track package, and projected gradient 
        ascent onto the given region will be performed.
        ----------------------------------------------------------------------------------------
        Output: waypoints for each mobile sensor, Shape= (num_time_steps,num_sensors,2)
    """

    step_size=max_linear_speed*dt
    p_trajs=[]
    
    q=q.reshape(-1,2)
    ps=ps.reshape(-1,2)
    
    for i in range(n_timesteps):
        # Calculate the gradient
        grad=f_dLdp(q=q,ps=ps)

        if np.any(np.isnan(grad)): # This is a hack that gets rid of degenerate gradient with random directions
                grad = np.random.random(grad.shape)
        
        grad=grad.reshape(-1,2)
        grad_sizes=np.linalg.norm(grad,axis=1)
        # print(ps,q)
        # print("grad_sizes",grad_sizes)

        grad_sizes[grad_sizes==0]=1 # Handle the case where the partial derivative is zero.

        update_steps=(grad.T/grad_sizes * step_size).T # Calculate the update steps to be applied to ps

        candid_ps=ps+update_steps # Calculate the direct update 
        
        # Perform the projection onto specified constraint region, if given.
        if not region is None:
            for j in range(len(ps)):
                proj=region.project_point(candid_ps[j,:])
                if not proj is None:
                    candid_ps[j,:]=proj
                    
        # Project candid_ps onto the "surveillance circle" once it steps into it
        if not np.all(np.linalg.norm(candid_ps-q,axis=1)>=epsilon):
            insiders=np.linalg.norm(candid_ps-q,axis=1)<epsilon
            ps=candid_ps
            ps[insiders]=polar_projection(q,candid_ps,epsilon)[insiders] # Update ps.
            p_trajs.append(ps)
            # break # Exit the loop once some mobile sensor's trajectory reaches the circle.
        else:
            ps=candid_ps # Update ps.
            p_trajs.append(ps)
    return np.array(p_trajs).reshape(-1,n_p,2) # Shape= (num_time_steps,num_sensors,2)
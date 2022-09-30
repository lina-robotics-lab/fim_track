import numpy as np
"""
Formation control based on artificial potential and virtual leaders.

The control algorithm is implemented referring to:

Leonard, N. E., & Fiorelli, E. (2001). Virtual leaders, artificial potentials and coordinated control of groups. Proceedings of the 40th IEEE Conference on Decision and Control (Cat. No.01CH37228), 3, 2968–2973. https://doi.org/10.1109/CDC.2001.980728

"""
def formation_control_force(robot_locs,robot_vels,alpha = 1, k=1,d0 = 1,d1=100,kD=0.2,max_force=1):
    """
    Given the robot locs and robot vels, compute the control force, f, to be applied on each robot. 
    NOte: we assume the update formula is to minimize the potential V, and f = grad(V). So dv/dt = -f and dx/dt=v.
    """
    # Use the outter product technique to get the pairwise displacement between sources and sensors
    p2p=robot_locs[:,np.newaxis]-robot_locs

    # calculate the pairwise distance, virtual leader and mobile sensor included.
    r=np.linalg.norm(p2p,axis=-1)


    def formation_control_force_mag(r):
        """
        r is expected to be a square symmetric matrix that contains the pair-wise distance between robots.
        r[i,j]=distance between i and j

        The potential function takes a more general form than the one in the following paper.

        Bachmayer, R., & Leonard, N. E. (2002). Vehicle networks for gradient descent in a sampled environment. Proceedings of the 41st IEEE Conference on Decision and Control, 2002., 1, 112–117. https://doi.org/10.1109/CDC.2002.1184477

        """
        # alpha = 0.01 # A smaller constant alpha gives a smaller overall field strength
        # k = 1
        # d0 = 0.3
        # d1 = 10
        F = np.zeros(r.shape)
        non_zero_entries = np.logical_not(r==0)
        F[non_zero_entries] = alpha*(1/r[non_zero_entries]-k*d0/r[non_zero_entries]**(k+1))
        F[r>d1]=0
        F[F>max_force] = max_force
        F[F<-max_force] = -max_force
        return F

    F=formation_control_force_mag(r) # Calculate inter-agent force magnitude
      
    non_zero_entries = np.logical_not(r==0)
    force_dir=np.zeros(p2p.shape)
    force_dir[non_zero_entries] = p2p[non_zero_entries]/r[non_zero_entries,np.newaxis]

    f = np.sum(force_dir*F[:,:,np.newaxis],axis=1)  # Sum over the F matrix to get the force on each individual robot.

    # kD = 0.9
    # Apply the dampening term
    f+= kD*robot_vels
    
    return f
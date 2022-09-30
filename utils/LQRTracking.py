import numpy as np
from utils.MotionGeneration import scaled_spline_motion


BURGER_MAX_LIN_VEL = 0.22*0.8
BURGER_MAX_ANG_VEL = 2.84
def LQR(As,Bs,Qs,Rs):
    n_state = As[0].shape[0]
    n_ref_motion = len(As)
    n_input = Bs[0].shape[1]
    
    Ps = np.zeros((n_state,n_state,n_ref_motion))
    Ks = np.zeros((n_input,n_state,n_ref_motion-1))

    P = Ps[:,:,n_ref_motion-1] = Qs[n_ref_motion-1]

    for i in range(n_ref_motion-2,-1,-1):
        B = Bs[i]
        A = As[i]
        Q = Qs[i]
        R = Rs[i]

        K = Ks[:,:,i]=np.linalg.inv(R+B.T.dot(P).dot(B)).dot(B.T).dot(P).dot(A)
        P = Ps[:,:,i] = Q + K.T.dot(R).dot(K) + (A-B.dot(K)).T.dot(P).dot(A-B.dot(K))
    
    return Ps,Ks

def regularize_angle(theta):
    """
        Convert an angle theta to [-pi,pi] representation.
    """
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.angle(cos+sin*1j)
 
def LQR_for_motion_mimicry(waypoints,planning_dt,x_0,Q,R,Vm = BURGER_MAX_LIN_VEL,Om=BURGER_MAX_ANG_VEL):
    """
        We use time-invariant Q, R matrice for the compuation of LQR.
    """
    #### Fit spline ########
    if len(waypoints)<=1:
        return [],[],[]

    # # Get rid of the waypoints that are left-behind.
    waypoints = waypoints[np.argmin(np.linalg.norm(waypoints-x_0[:2],axis=1)):]
    p,theta,v,omega,dsdt=scaled_spline_motion(waypoints,planning_dt,Vm,Om)
    
    if len(p)==0 or len(theta)==0:
        return [],[],[]

    ref_x = np.concatenate([p,theta.reshape(-1,1)],axis=1)

    ref_u = np.array([v,omega]).T
    
    #### Prepare for LQR Backward Pass #######
    n_state = ref_x.shape[1]
    n_input = ref_u.shape[1]
    n_ref_motion=len(p)
    I = np.eye(n_state)
    def tank_drive_A(v,theta,planning_dt):
        A = I + \
            np.array([
                [0, 0, -v*np.sin(theta)],
                [0,0, v*np.cos(theta)],
                [0,0,0]
            ])* planning_dt
        return A
    def tank_drive_B(theta,planning_dt):
        B = np.array([
            [np.cos(theta),0],
            [np.sin(theta),0],
            [0,1]
        ])*planning_dt
        return B
    
    
    As = [ tank_drive_A(v[k],theta[k],planning_dt) for k in range(n_ref_motion)]
    Bs = [ tank_drive_B(theta[k],planning_dt) for k in range(n_ref_motion)]
    
    Qs = [Q for i in range(n_ref_motion)]
    Rs = [R for i in range(n_ref_motion-1)]
    
    Ps,Ks=LQR(As,Bs,Qs,Rs)
  
    ################## LQR Forward pass for 2D tank drive ##################
    # Initial deviation
    dx_0 = x_0-ref_x[0]
    dx_0[-1]=regularize_angle(dx_0[-1])
    
    # Data containers
    xhat=np.zeros(ref_x.shape)
    uhat = np.zeros(ref_u.shape)
    dx=dx_0
    
    for i in range(n_ref_motion-1):     
        dx[-1]=regularize_angle(dx[-1])
        x = ref_x[i] + dx 
        x[-1]=regularize_angle(x[-1])
        xhat[i]=x
    
        du = -Ks[:,:,i].dot(dx)
        uhat[i,:]=(ref_u[i]+du)
        
        dx = As[i].dot(dx)+Bs[i].dot(du)
    
    
    return uhat,xhat,p
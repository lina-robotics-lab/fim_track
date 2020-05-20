import numpy as np

def fit_spatial_polynomial(waypoints,poly_order, state_space_dim):
    """
        Fit a spatial polynomial p(s)-> R^state_space_dim, s in 0~1, to fit the waypoints.
    """
    if waypoints.shape[1]!=state_space_dim:
        waypoints=waypoints.T
    
    assert(waypoints.shape[1]==state_space_dim)
    
    n = waypoints.shape[0]

    s = np.array([i/(n-1) for i in range(n)])
    S = np.vstack([np.power(s,k) for k in range(poly_order+1)])
    S = S.T
    
    # The two formulas below are equivalent if S is full rank.
#     poly_coefs= np.linalg.inv(S.dot(S.T))).dot(waypoints)
    poly_coefs = np.linalg.pinv(S).dot(waypoints)
    return poly_coefs,S

def polynomial(poly_coefs,x):
    '''
        Evaluate the value of the polynomial specified by poly_coefs at locations x.
    '''
    S = np.vstack([np.power(x,k) for k in range(len(poly_coefs))])
    y = np.array(poly_coefs).dot(S)
    return y
def diff_poly_coefs(poly_coefs):
    '''
        Calculate the coefs of the polynomial after taking the first-order derivative.
    '''
    if len(poly_coefs)==1:
        coefs = [0]
    else:
        coefs = np.array(range(len(poly_coefs)))*poly_coefs
        coefs = coefs[1:]
    return coefs

def generate_2D_spline_motion(waypoints,poly_order, state_space_dim):
    coef,S = fit_spatial_polynomial(waypoints,poly_order, state_space_dim)
    
    # coef.shape = (poly_order+1,state_space_dim)
    # S.shape = (n_waypoints,poly_order+1), 
    # S = [[1,s_i,s_i^2,s_i^3,...,s_i^poly_order]]_{i=0...n_waypoints}, s_i = i/n_waypoints
    
    dotCoef = np.vstack([diff_poly_coefs(coef[:,i]) for i in range(state_space_dim)]).T
    # dotCoef.shape = (poly_order,state_space_dim)
    
    ddotCoef = np.vstack([diff_poly_coefs(dotCoef[:,i]) for i in range(state_space_dim)]).T
    # ddotCoef.shape = (poly_order-1,state_space_dim)
    
    p = S[:,:poly_order+1].dot(coef)
    # p.shape = (n_waypoints,state_space_dim)
    
    pDot = S[:,:poly_order].dot(dotCoef)
    # pDot.shape = (n_waypoints,state_space_dim)
    
    pDDot = S[:,:poly_order-1].dot(ddotCoef)
    # pDot.shape = (n_waypoints,state_space_dim)
    
    theta = np.arctan2(pDot[:,1],pDot[:,0])
    # The facing angles at each p, shape=(n_waypoints,)
    
    v= np.linalg.norm(pDot,axis=1)
    # The velocity, derivative in s. shape = (n_waypoints,)
    
    omega = (pDDot[:,1]*pDot[:,0]-pDDot[:,0]*pDot[:,1])/np.power(v,2)
    # The angular velocity, rotating counter-clockwise as positive. shape=(n_waypoints,)
    return p,pDot,pDDot,theta,v,omega
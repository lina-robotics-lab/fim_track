import numpy as np

BURGER_MAX_LIN_VEL = 0.22*0.8
BURGER_MAX_ANG_VEL = 2.84

def unscaled_spline_motion(waypoints,poly_order, space_dim,n_output):
    
    # Local Helper Functions
    def fit_spatial_polynomial(waypoints,poly_order, space_dim):
        """
            Fit a spatial polynomial p(s)-> R^space_dim, s in 0~1, to fit the waypoints.
        """
        if waypoints.shape[1]!=space_dim:
            waypoints=waypoints.T

        assert(waypoints.shape[1]==space_dim)

        n = waypoints.shape[0]
        if n<=1:
            return []

        s = np.array([i/(n-1) for i in range(n)])
        S = np.vstack([np.power(s,k) for k in range(poly_order+1)])
        S = S.T

        # The two formulas below are equivalent if S is full rank.
    #     poly_coefs= np.linalg.inv(S.dot(S.T))).dot(waypoints)
        poly_coefs = np.linalg.pinv(S).dot(waypoints)
        return poly_coefs

    # A debug-purpose function.
    # def polynomial(poly_coefs,x):
    #     '''
    #         Evaluate the value of the polynomial specified by poly_coefs at locations x.
    #     '''
    #     S = np.vstack([np.power(x,k) for k in range(len(poly_coefs))])
    #     y = np.array(poly_coefs).dot(S)
    #     return y

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
    ######### End of Helper Functions #################################
    if n_output <=1:
        return np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
    coef = fit_spatial_polynomial(waypoints,poly_order, space_dim)
    s = np.array([i/(n_output-1) for i in range(n_output)])
    S = np.vstack([np.power(s,k) for k in range(poly_order+1)])
    S=S.T

    # coef.shape = (poly_order+1,space_dim)
    # S.shape = (n_waypoints,poly_order+1), 
    # S = [[1,s_i,s_i^2,s_i^3,...,s_i^poly_order]]_{i=0...n_output-1}, s_i = i/(n_output-1)
    
    dotCoef = np.vstack([diff_poly_coefs(coef[:,i]) for i in range(space_dim)]).T
    # dotCoef.shape = (poly_order,space_dim)
    
    ddotCoef = np.vstack([diff_poly_coefs(dotCoef[:,i]) for i in range(space_dim)]).T
    # ddotCoef.shape = (poly_order-1,space_dim)
    
    p = S[:,:poly_order+1].dot(coef)
    # p.shape = (n_waypoints,space_dim)
    
    pDot = S[:,:poly_order].dot(dotCoef)
    # pDot.shape = (n_waypoints,space_dim)
    
    pDDot = S[:,:poly_order-1].dot(ddotCoef)
    # pDot.shape = (n_waypoints,space_dim)
    
    theta = np.arctan2(pDot[:,1],pDot[:,0])
    # The facing angles at each p, shape=(n_waypoints,)
    
    v= np.linalg.norm(pDot,axis=1)
    # The velocity, derivative in s. shape = (n_waypoints,)
    
    omega = (pDDot[:,1]*pDot[:,0]-pDDot[:,0]*pDot[:,1])/np.power(v,2)
    # The angular velocity, rotating counter-clockwise as positive. shape=(n_waypoints,)
    return p,pDot,pDDot,theta,v,omega
def scaled_spline_motion(waypoints,planning_dt,poly_order=3,space_dim=2):
    """
        The synchronized max uniform speed scheduling.
        space_dim: the dimension of space. Normally 2 or 3.
    """
    Vm = BURGER_MAX_LIN_VEL
    Om = BURGER_MAX_ANG_VEL
    
    
    # Prepare the data for calculating nstar
    n_waypoints=len(waypoints)
    if len(waypoints)<=1:
        return [],[],[],[],[]
    N=np.max([100,4*n_waypoints]) 
    # Heuristic choice. The number of grid points to be used in grid_search for determining nstar.
    p,pDot,pDDot,theta,v,omega = unscaled_spline_motion(waypoints,poly_order, space_dim,N)
    
    
    # Calculate nstar
    m = np.min([Vm/np.abs(v),Om/np.abs(omega)],axis=0)
    mstar=np.min(m)
    nstar = int(np.ceil(1/(mstar*planning_dt)))

    
    dsdt =  1/(nstar*planning_dt)
    p,pDot,pDDot,theta,v,omega = unscaled_spline_motion(waypoints,poly_order, space_dim,nstar)
    
    v*=dsdt
    omega*=dsdt
#   dsdt is the scaling factor to be multiplied on v and omega, so that they do not exceed the maximal velocity limit.
    
    
    return p,theta,v,omega,dsdt
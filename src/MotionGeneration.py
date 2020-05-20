import numpy as np
def fit_spatial_polynomial(waypoints,poly_order, state_space_dim):
    if waypoints.shape[1]!=state_space_dim:
        waypoints=waypoints.T
    
    assert(waypoints.shape[1]==state_space_dim)
    
    n = waypoints.shape[0]

    # s = [0,1/(n-1),2/(n-1),...,1]
    s = np.array([i/(n-1) for i in range(n)])
    
    # S = [ [1 s0 s0^2 s0^3 ... s0^poly_order]
    #       [1 s1 s1^2 s1^3 ... s1^poly_order]
    #        .
    #        .
    #        .
    #       [1 sn sn^2 sn^3 ... sn^poly_order]]
    S = np.vstack([np.power(s,k) for k in range(poly_order+1)])
    S = S.T
    
    # The two formulas below are equivalent if S is full rank.
#     poly_coefs= np.linalg.inv(S.dot(S.T))).dot(waypoints)
    poly_coefs = np.linalg.pinv(S).dot(waypoints)
    return poly_coefs,S
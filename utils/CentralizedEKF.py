import numpy as np
class CentralizedEKF:
    """
        The centralized EKF.

        The estimator assumes a constant-velocity source movement model, with the velocity to be the fundamental uncertainty.
    """
    def __init__(self,q_0,R_mag=1,Q_mag=1):
        '''
        q_0 should be a vector, the initial guess of source position.
        
        R_mag,Q_mag, should be scalars.
        '''
        
        self.q = q_0 # Mean
        self.qdot = np.zeros(self.q.shape) # Velocity Mean
        
        self.z = np.hstack([self.q,self.qdot]) # The source predicted state, public variable.
        self._zbar = np.hstack([self.q,self.qdot]) # The source corrected state, private variable.
        
        
        
        self.qdim = len(self.q)
        
        self.P = np.eye(len(self.z)) # Covariance of [q,qdot]. Initialized to be the identity matrix
        
        self.R_mag = R_mag 
        self.Q_mag = Q_mag
       
    def dfdz(self,z):
        n = len(z)//2
        O =np.zeros((n,n))
        I=np.eye(n)
        return np.vstack([np.hstack([I,I]),np.hstack([O,I])])

    def f(self,z):
        """
            The constant velocity model.
        """
        A = self.dfdz(z)
        
        return A.dot(z)
    def update(self,h,dhdz,y,p):
        """
        h is a function handle h(z,p), the measurement function that maps z,p to y.
        dhdz(z,p) is its derivative function handle.
        
        y is the actual measurement value, subject to measurement noise.
        
        p is the positions of the robots.
        
        z_neighbor is the list of z estimations collected from the neighbors(including self).
        """
        A = self.dfdz(self._zbar)
        C = dhdz(self.z,p)
        R = self.R_mag*np.eye(len(y))
        Q = self.Q_mag*np.eye(len(self.z))
        
        # The Kalman Gain
        K = A.dot(self.P).dot(C.T).dot(    np.linalg.inv(C.dot(self.P).dot(C.T)+R)      )

        # Mean and covariance update

        self.z = self.f(self.z)+K.dot(y-h(self.z,p))                                 
        self.P = A.dot(self.P).dot(A.T)+ Q- K.dot(C.dot(self.P).dot(C.T)+R).dot(K.T)
        
    def update_and_estimate_loc(self,h,dhdz,y,p):
        if not np.any(y == np.inf):
            self.update(h,dhdz,y,p)
        self.q = self.z[:len(self.q)]
        
        return self.q

    
        
        
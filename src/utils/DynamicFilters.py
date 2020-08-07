'''
jax is a python package implemented by Google, for the ease of auto-differentiation.

jax wraps all the numpy functions to be ready for auto-differentiation, and package them in jax.numpy

Every function written using the jnp wrappers can be differentiated using the grad or jac function.
'''
import numpy as np
import jax.numpy as jnp
from jax import jacfwd, jit,grad

'''
 We will borrow the filterpy package as our EKF utility. Reference to be added later
'''
from filterpy.kalman import ExtendedKalmanFilter as EKF
from functools import partial
'''
Use self-built particle filter package
'''
from utils.ParticleFilterBasic import ParticleFilterBasic as PF


def single_meas_func(C1,C0,k,b,dist):
	return k*jnp.power(dist-C1,b)+C0


def joint_meas_func(C1s,C0s,ks,bs,x,ps):

	# Casting for the compatibility of jax.numpy
	
	C1s=jnp.array(C1s)
	C0s=jnp.array(C0s)
	ks=jnp.array(ks)
	bs=jnp.array(bs)
	ps=jnp.array(ps)

	# Keep in mind that x is a vector of [q,q'], thus only the first half of components are observable.    
	dists=jnp.linalg.norm(x[:len(x)//2]-ps,axis=1)

	return single_meas_func(C1s,C0s,ks,bs,dists) 
	
def getDynamicFilter(num_sensors,num_targets,C1s,C0s,ks,bs,initial_guess=None,filterType='ekf'):
	"""
		We assume the coefficients for each mobile sensors can be different, so the
		coefs passed in are in plural form.
	"""
	if None in C1s or None in C0s or None in ks or None in bs:
		print('Coefficients not fully yet received.')
		return None


	meas_func=partial(joint_meas_func,C1s,C0s,ks,bs)# Freeze the coefficients, the signature becomes meas_func(x,ps)
	return TargetTrackingSS(num_sensors,num_targets,meas_func,initial_guess=initial_guess,filterType=filterType)


class TargetTrackingSS:
	'''
		The state space model for the target tracking system is the constant velocity model.

		It assumes the target always move in constant velocity, and that is good enough for 
		many location estimation tasks.
		
		The states are in the format of:
			x=[x1,x2,...,xn,x1',x2',...,xn']
		
		The system dynamics becomes:    
			
			x'=Ax, y=meas_func(x,p)
			
			We ignore the sampling time constant and simplify the update formula as
			
				x_i(t+1) = x_i(t) + x_i'(t) + d
				x_i'(t) = x_i'(t) + w
		
			so 	   A =[[I,I],
					   [O,I]]
				it has shape 4n x 4n, and each block matrix has shape 2n x 2n

				  
			and meas_func takes in a num_targets x 2 vector x, and a num_sensors x 2 vector p.
			The jacobian will be automatically calculated as dmeas_func/dx only. 
	'''
	def __init__(self, num_sensors,num_targets,meas_func, initial_guess=None, filterType='ekf'):
            self.num_sensors=num_sensors
            self.num_targets=num_targets
            self.filterType=filterType

            n=2*self.num_targets # For each target, its state is a 4D vector, including its position and the derivative of its position, both in 2D of course.
            O=np.zeros((n,n))
            I=np.eye(n)
            self.A=np.vstack([np.hstack([I,I]),np.hstack([O,I])])

            self.meas_func=meas_func

            self.ps=np.zeros((self.num_sensors,2))

            # Initialize the filter object
            if filterType=='ekf':
                self.filter=EKF(dim_x=4*num_targets,dim_z=num_sensors) # For each sensor there will be one scalar reading. So the dimension of output z is num_sensors.
                self.filter.F=self.A # F is the state transition matrix.
                # self.filter.Q = np.eye(4*num_targets) * 1 # Process noise matrix
                # self.filter.R = np.eye(num_sensors) * 1 # Measurement noise matrix
            elif filterType=='pf':
                self.filter=PF(dim_x=4*num_targets, dim_z=num_sensors, sensor_std = 0.5, move_std = 0.1, N = 50)
            else:
                self.filter=None
                print('{} is not yet supported'.format(filterType))

            if initial_guess is None:
                    self.filter.x=np.zeros(self.num_targets*4)
            else:
                    if filterType=='ekf':
                            self.filter.x=np.pad(initial_guess,(0,4-len(initial_guess)),'constant',constant_values=0)
                    elif filterType=='pf':
                            padded = np.pad(initial_guess,(0,4-len(initial_guess)),'constant',constant_values=0)
                            self.filter.init_gaussian(padded, np.ones(4*num_targets)*10)

	
	def update_and_estimate_loc(self,ps,meas):
	    
	    self.update_filters(ps,meas)

	    return self.current_state_corrected()[:2]


	def current_state_corrected(self):
		return self.filter.x_post
	
	def current_state_uncorrected(self):
		return self.filter.x_prior
	
	def update_filters(self,new_ps, new_measurements):
            self.update_ps_(new_ps)
            #self.filter.update(new_measurements,self.dhxdx,self.hx)
            if self.filterType=='ekf':
                self.filter.update(new_measurements,self.dhxdx,self.hx)
            elif self.filterType=='pf':
                self.filter.update(new_measurements,new_ps, self.hx)
            else:
                self.filter.update(new_measurements,self.dhxdx,self.hx)

            self.filter.predict()
	
	def update_ps_(self,new_ps): 
		'''
		A private function, used only when new sensor locations are passed in during the filter update
		'''
		self.ps=new_ps
		
	def hx(self,x):
		return self.meas_func(x,self.ps)
	
	# Use jax to automatically compute the jacobian of measurement function, with respect to state only.
	def dhxdx(self,x):
		return jacfwd(self.hx)(x)
	
	

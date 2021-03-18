"""
Both analytic_dLdp function and dLdp function can be used to caculate the value
of dLdp. 

They yield the same result given the same inputs, only that dLdp relies on jax library to do the
auto-differentiation.
"""

import numpy as np
from utils.MutualSepPathPlanning import sep_func


def dhdr(r,C1s,C0s,ks,bs):
    return ks*bs*(r-C1s)**(bs-1)
def d2hdr2(r,C1s,C0s,ks,bs):
    return dhdr(r,C1s,C0s,ks,bs)*(bs-1)/(r-C1s)


def analytic_dhdz(x,ps,C1s,C0s,ks,bs):

    q = x.flatten()
    q = q[:len(q)//2]
    dhdq = analytic_dhdq(q,ps,C1s,C0s,ks,bs)
    return jnp.hstack([dhdq,np.zeros(dhdq.shape)])

def analytic_dhdq(q,ps,C1s,C0s,ks,bs):
    rs = jnp.linalg.norm(ps-q,axis=1)
   
    r_hat = ((ps-q).T/rs).T
    d = dhdr(rs,C1s,C0s,ks,bs)
    dhdq=-(d * r_hat.T).T
    return dhdq
def analytic_FIM(q,ps,C1s,C0s,ks,bs):
    # rs = np.linalg.norm(ps-q,axis=1)
    rs = jnp.linalg.norm(ps-q,axis=1)
    r_hat = ((ps-q).T/rs).T


    d = dhdr(rs,C1s,C0s,ks,bs)
    dd = d2hdr2(rs,C1s,C0s,ks,bs)       

    As = (-d*r_hat.T).T

    return As.T.dot(As) # Current FIM

def analytic_dLdp(q,ps,C1s,C0s,ks,bs,FIM=None):
    """
        The gradient is taken with respect to all the ps passed in. 

        The FIM is by default calculated internally, but if it is passed in, will
        use the passed in FIM for the calculation of Q below.
    """
  
    rs = np.linalg.norm(ps-q,axis=1)
    r_hat = ((ps-q).T/rs).T
    t_hat=np.zeros(r_hat.shape)
    t_hat[:,0]=-r_hat[:,1]
    t_hat[:,1]=r_hat[:,0]

    d = dhdr(rs,C1s,C0s,ks,bs)
    dd = d2hdr2(rs,C1s,C0s,ks,bs)

    wrhat=(d*r_hat.T).T

    if FIM is None:
        Q = np.linalg.inv(wrhat.T.dot(wrhat)) # Default calculation of FIM^-1
    else:
        # print('Coordinating')
        if np.linalg.matrix_rank(FIM) < 2:
            FIM = FIM + 1e-9*np.eye(2)
        Q = np.linalg.inv(FIM) # Using the passed in FIM.

    c1 = -2*d*dd*np.linalg.norm(Q.dot(r_hat.T),axis=0)**2
    c2 = -2*(1/rs)*(d**2)*np.einsum('ij,ij->j',Q.dot(r_hat.T),Q.dot(t_hat.T))

    return (c1*r_hat.T+c2*t_hat.T).T


from jax import grad,jit, jacfwd
from matplotlib import pyplot as plt
import jax.numpy as jnp

from functools import partial

def single_meas_func(C1,C0,k,b,dist):
    """
        The small h function, for each individual measurement.
    """
    return k*jnp.power(dist-C1,b)+C0


def joint_meas_func(C1s,C0s,ks,bs,q,ps):
    """
        The big H function, the array of all individual measurements.
    """

    # Casting for the compatibility of jax.numpy

    C1s=jnp.array(C1s)
    C0s=jnp.array(C0s)
    ks=jnp.array(ks)
    bs=jnp.array(bs)
    ps=jnp.array(ps)

    # Keep in mind that x is a vector of [q,q'], thus only the first half of components are observable.    
    
    dists=jnp.linalg.norm(q-ps,axis=1)

    return single_meas_func(C1s,C0s,ks,bs,dists) 


def FIM(C1s,C0s,ks,bs,sigma=1):
    """
       The computation of Fish Information Matrix, using definition.
    """
    
    H=partial(joint_meas_func, C1s,C0s,ks,bs)
    
    # Taking partial derivative of H w.r.t. the zeroth argument, which is q.
    dHdq=jit(jacfwd(H,argnums=0))
    # import pdb
    # pdb.set_trace()
    return lambda q,ps:1/(jnp.power(sigma,2)) *  dHdq(q.reshape(ps.shape[1],),ps).T.dot(dHdq(q.reshape(ps.shape[1],),ps))

def FIM_mix(C1s,C0s,ks,bs,sigma=1):

    """
       The computation of Fish Information Matrix mixture, with different q for different agents.
    """

    H=partial(joint_meas_func, C1s,C0s,ks,bs)

    # Taking partial derivative of H w.r.t. the zeroth argument, which is q.
    dHdq=jit(jacfwd(H,argnums=0)) # dHdq expects (q,ps) with q being a single vector, and outputs a [len(ps) x dim(q)] Jacobian matrix. 

    dHdq_mix = lambda qs,ps: jnp.ones(len(ps)).dot(dHdq(qs,ps)) 
   
    # If we passed in a stack of vectors qs to dHdq, it will output a [len(ps) x shape(qs)[0] x shape(qs)[1]] Jacobian tensor, with many independent coord. taking zero values. We can reshape the matrix
    # back as [len(ps)x dim(q)] by left-multiplying an all-one row vector

    return lambda qs,ps:1/(jnp.power(sigma,2)) *  dHdq_mix(qs,ps).T.dot(dHdq_mix(qs,ps))

def L(C1s,C0s,ks,bs,sigma=1):
    """
        The loss function big L. 
    """
    # return jnp.linalg.det(FIM(q,ps,C1s,C0s,ks,bs,sigma))
    return lambda q,ps:jnp.trace(jnp.linalg.inv(FIM(C1s,C0s,ks,bs,sigma)(q,ps)))

def dAinv(inv_A,dAdp):
    # import pdb
    # pdb.set_trace()
    return -inv_A.dot(dAdp.transpose((2,3,0,1)).dot(inv_A)).transpose((3,0,1,2))

def dLdp(C1s,C0s,ks,bs,sigma=1,FIM_func=FIM):

    """
        The dLdp caculation using jacfwd.
    """
    # return np.array(jit(jacfwd(L,argnums=1))(q,ps,C1s,C0s,ks,bs,sigma))
   
    # A = FIM(q,ps,C1s,C0s,ks,bs,sigma)
    
    # Construct A(q,ps)
    A = FIM_func(C1s,C0s,ks,bs,sigma)

    # Construct dAdp(q,ps)
    dAdp = jit(jacfwd(A,argnums=1))
    
    # Construct inv_A(q,ps)
    inv_A=lambda q,ps: jnp.linalg.inv(A(q,ps))
    
    # print(np.trace(-dAinv(inv_A,dAdp),axis1=0,axis2=1)-np.array(jit(jacfwd(L,argnums=1))(q,ps,C1s,C0s,ks,bs,sigma)))
    
    # Construct dLdP(q,ps)
    return lambda q,ps: np.array(jnp.trace(dAinv(inv_A(q,ps),dAdp(q,ps)),axis1=0,axis2=1)) 

def dSdp(C1s,C0s,ks,bs,sigma=1):

    """
        S = L^1/b
        dSdp = 1/b * L^(1/b-1) * dLdp
    """
    bs = np.array(bs)
    dL = dLdp(C1s,C0s,ks,bs,sigma)
    f_L = L(C1s,C0s,ks,bs,sigma)

    # import pdb
    
    # pdb.set_trace()
    return lambda q,ps: np.abs((1/bs * f_L(q,ps)**(1/bs-1)).reshape(len(bs),1)) * dL(q,ps)

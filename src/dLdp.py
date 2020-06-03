"""
Both analytic_dLdp function and dLdp function can be used to caculate the value
of dLdp. 

They yield the same result given the same inputs, only that dLdp relies on jax library to do the
auto-differentiation.
"""

import numpy as np
def analytic_L(q,ps,C1s,C0s,ks,bs,sigma=1):
    
    n_p=len(ps)
    r=jnp.linalg.norm(ps-q,axis=1).reshape(-1,1)
    r_hat=(ps-q)/r

    L=0
    for i in range(n_p):
        for j in range(n_p):
                
            rkrj=jnp.min([r_hat[i,:].dot(r_hat[j,:]),1])
            
            L+=(bs[i]*bs[j]*ks[i]*ks[j])**2 * (r[i]-C1s[i])**(2*bs[i]-2) * (r[j]-C1s[j])**(2*bs[j]-2) * (1-rkrj**2)
            
    L/=2*sigma**2
    
    return L[0]
def analytic_dLdp(q,ps,C1s,C0s,ks,bs,sigma=1):
    """
        Typically the sigma value is just a scaling on the magnitude of the gradient, so it can take a default value=1.
    """
    n_p=len(ps)
    r=np.linalg.norm(ps-q,axis=1).reshape(-1,1)
    r_hat=(ps-q)/r
    t_hat=np.zeros(r_hat.shape)
    t_hat[:,0]=-r_hat[:,1]
    t_hat[:,1]=r_hat[:,0]

    dLdeta=np.zeros(n_p).reshape(-1,1)
    dLdr=np.zeros(n_p).reshape(-1,1)


    for i in range(n_p):
        Keta=2*(ks[i]*bs[i])**2/(sigma**2) * (r[i]-C1s[i])**(2*bs[i]-2)
        Kr=2*(ks[i]*bs[i])**2/(sigma**2) * (bs[i]-1) * (r[i]-C1s[i])**(2*bs[i]-3)
        sum_eta=sum_kr=0
        for j in range(n_p):
                
            rkrj=np.max([np.min([r_hat[i,:].dot(r_hat[j,:]),1]),-1])
            
            direction=np.sign(np.linalg.det(r_hat[[j,i],:]))

            sum_eta += (ks[j]*bs[j])**2 * (r[j]-C1s[j])**(2*bs[j]-2) * rkrj * np.sqrt(1-rkrj**2) * direction
            sum_kr += (ks[j]*bs[j])**2 * (r[j]-C1s[j])**(2*bs[j]-2) * (1-rkrj**2)
        
        dLdeta[i]=Keta*sum_eta
        dLdr[i]=Kr*sum_kr
        
    dLdp = dLdr * r_hat  + (dLdeta/r) * t_hat
    
    
    return dLdp


#### Temporaily Freeze the jax-dependent part. The analytic version has less dependency thus more compatibility.###################################################################


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


def FIM(q,ps,C1s,C0s,ks,bs,sigma=1):
    """
       The computation of Fish Information Matrix, using definition.
    """
    
    H=partial(joint_meas_func, C1s,C0s,ks,bs)
    q=q.reshape(ps.shape[1],)
    # Taking partial derivative of H w.r.t. the zeroth argument, which is q.
    dHdq=jit(jacfwd(H,argnums=0))
    # import pdb
    # pdb.set_trace()
    return 1/(jnp.power(sigma,2)) *  dHdq(q,ps).T.dot(dHdq(q,ps))

def L(q,ps,C1s,C0s,ks,bs,sigma=1):
    """
        The reward function big L. It is just det(FIM)
    """
    
    # return jnp.linalg.det(FIM(q,ps,C1s,C0s,ks,bs,sigma))
    return -jnp.trace(jnp.linalg.inv(FIM(q,ps,C1s,C0s,ks,bs,sigma)))

def dAinv(inv_A,dAdp):
    return -inv_A.dot(dAdp.transpose(2,3,0,1).dot(inv_A)).transpose(3,0,1,2)

def dLdp(q,ps,C1s,C0s,ks,bs,sigma=1):

    """
        The dLdp caculation using jacfwd.
    """
    # return np.array(jit(jacfwd(L,argnums=1))(q,ps,C1s,C0s,ks,bs,sigma))
   
    A = FIM(q,ps,C1s,C0s,ks,bs,sigma)
    inv_A=np.linalg.inv(A)
    
    dAdp = np.array(jit(jacfwd(FIM,argnums=1))(q,ps,C1s,C0s,ks,bs,sigma))
    
    # print(np.trace(-dAinv(inv_A,dAdp),axis1=0,axis2=1)-np.array(jit(jacfwd(L,argnums=1))(q,ps,C1s,C0s,ks,bs,sigma)))
    
    return np.trace(-dAinv(inv_A,dAdp),axis1=0,axis2=1)
# Import packages.
import cvxpy as cp
import numpy as np
from utils.dLdp import dhdr,d2hdr2,analytic_FIM

def trust_region_plan_single_step(q,ps,C1s,C0s,ks,bs,bounding_radius = 1):
    '''
    bounding_radius: The physical distance reachable in one time step by the sensors' ability.
    '''
    # Two important helper functions for the trust region
    def trust_region_adjust(prob,step_size,new_p,new_z,new_Fp,p_iter,z_iter,Fp_iter):
        
        
        
        cons = [c.value for c in prob.constraints]
        cons.append(np.all(np.linalg.norm(new_p-ps,axis=1)<=bounding_radius))
        not_violated=np.all(cons)
        
        improved = np.min(np.linalg.eigvals(new_Fp))>np.min(np.linalg.eigvals(Fp_iter))
        
        if (not_violated and improved) or prob.status!='optimal': # To be implemented later
            new_step_size = np.min([step_size*2,bounding_radius])
            return new_p,new_z,new_step_size
        else:
            new_step_size = step_size/2
            return p_iter,z_iter,new_step_size
    
    def solve_linearized_problem(step_size,bounding_radius,p_init,p_iter,z_iter,C1s,C0s,ks,bs):
        # Prepare the raw data
        rs = np.linalg.norm(p_iter-q,axis=1)
        r_hat = ((p_iter-q).T/rs).T
        t_hat=np.zeros(r_hat.shape)
        t_hat[:,0]=-r_hat[:,1]
        t_hat[:,1]=r_hat[:,0]

        d = dhdr(rs,C1s,C0s,ks,bs)
        dd = d2hdr2(rs,C1s,C0s,ks,bs)       

        As = (-d*r_hat.T).T

        Fp=As.T.dot(As) # Current FIM

        # Declare the optimization-related variables.
        dps = cp.Variable(p_iter.shape)
        dz = cp.Variable(nonneg=True)

        # Prepare the matrix directional derivative DF[dps] matrix.
        rhat_conj=[r_hat[i,:,np.newaxis].dot(r_hat[i,np.newaxis,:]) for i in range(len(r_hat))]
        that_conj = [t_hat[i,:,np.newaxis].dot(t_hat[i,np.newaxis,:]) for i in range(len(t_hat))]
        dpdAs =[-dd*rc-1/r*d*tc for dd,d,r,rc,tc in zip(dd,d,rs,rhat_conj,that_conj)]

        dps_As = [dps.T[:,i,np.newaxis]@(As[i,np.newaxis,:]) for i in range(len(As))]

        half_DF = np.hstack(dpdAs)@(cp.vstack(dps_As))

        DF = half_DF+half_DF.T

        # Prepare the objective and constraints
        I = np.eye(p_iter.shape[1])
        constraints = [Fp-z_iter*I+DF-dz*I>>0,
                       cp.norm(p_iter+dps-p_init,axis=1)<=bounding_radius,
                       cp.norm(dps,axis=1)<=step_size]
        prob = cp.Problem(cp.Minimize(-dz),constraints)
        prob.solve()
        if prob.status == 'optimal':
            newFp = analytic_FIM(q,p_iter+dps.value,C1s,C0s,ks,bs)
            return p_iter+dps.value,newFp,z_iter+prob.value,prob
        else:
            return p_iter,Fp,z_iter,prob
           
      
    # The initial z should take value zero.
    z_iter = 0
    # The initial p should take the value of current ps
    p_init = np.array(ps)
    p_iter = np.array(p_init)

    # The initial step size is set as the same as bounding_radius
    step_size = bounding_radius

    # Enter the inner loop of solving a sequence of linearized SDP
    for i in range(100):
        Fp_iter = analytic_FIM(q,p_iter,C1s,C0s,ks,bs)
        
        new_p,new_Fp,new_z,prob = solve_linearized_problem(step_size,bounding_radius,p_init,p_iter,z_iter,C1s,C0s,ks,bs)
        
        p_iter,z_iter,step_size=trust_region_adjust(prob,step_size,new_p,new_z,new_Fp,p_iter,z_iter,Fp_iter) # Adjust the iteration parameters
        
        assert( np.all([c.value for c in prob.constraints]))
        if step_size<1e-4: # Optimality certificate
            break
    return p_iter
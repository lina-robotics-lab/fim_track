import numpy as np
def grad_est(ps,y,p0=np.array([0,0])):
    """
    The gradient estimation algorithm follows from Lemma 5.1 in

    Ogren, P., Fiorelli, E., & Leonard, N. E. (2004). Cooperative Control of Mobile Sensor Networks: Adaptive Gradient Climbing in a Distributed Environment. IEEE Transactions on Automatic Control, 49(8), 1292–1302. https://doi.org/10.1109/TAC.2004.832203

    Return the best estimation of [grad_p(F)(p0),F(p0)] given ps, y.
    """
    p0 = np.array([0,1])
    one = np.ones(len(ps))
    one = one[:,np.newaxis]
    C = np.hstack([ps-p0,one])
    est = np.linalg.pinv(C).dot(y) # = (grad(p0),function_at(p0)). In fact, changing p0 only changes the estimate of function_at(p0), but grad(p0) will be the same.
    grad = est[:len(p0)]
    val = est[len(p0)]
    return grad,val
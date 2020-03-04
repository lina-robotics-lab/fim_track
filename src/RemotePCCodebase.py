import numpy as np
from sklearn.linear_model import LinearRegression
def top_n_mean(readings,n):
    rowwise_sort=np.sort(readings,axis=1)
    return np.mean(rowwise_sort[:,-n:],axis=1)

def loss(C_1,dists,light_strengths,C_0=0,fit_type='light_readings',loss_type='rmse'):
    '''
        h(r)=k(r-C_1)**b+C_0
    '''
    
    x=np.log(dists-C_1).reshape(-1,1)
    y=np.log(light_strengths-C_0).reshape(-1,1)

    model=LinearRegression().fit(x,y)

    k=np.exp(model.intercept_[0])
    b=model.coef_[0][0]


    if fit_type=="light_readings":
        ## h(r)=k(r-C_1)**b+C_0
        yhat=k*(dists-C_1)**b+C_0
        
        if loss_type=='max':
            e=np.sqrt(np.max((yhat-light_strengths)**2))
        else:
            e=np.sqrt(np.mean((yhat-light_strengths)**2))
    elif fit_type=='dists':
        rhat=rhat(light_strengths,C_1,C_0,k,b)
        if loss_type=='max':
            e=np.sqrt(np.max((rhat-dists)**2))
        else:
            e=np.sqrt(np.mean((rhat-dists)**2))
    
    return e,C_1,C_0,k,b


## The once and for all parameter calibration function.
def calibrate_meas_coef(robot_loc,target_loc,light_readings,fit_type='light_readings',loss_type='rmse'):
    dists=np.sqrt(np.sum((robot_loc-target_loc)**2,axis=1))
    light_strengths=top_n_mean(light_readings,2)
    
    ls=[]
    ks=[]
    bs=[]
    C_1s= np.linspace(-1,np.min(dists)-0.01,100)
    C_0s=np.linspace(-1,np.min(light_strengths)-0.01,100)
    ls=[]
    mls=[]
    for C_1 in C_1s:
        for C_0 in C_0s:
            
            l,C_1,C_0,k,b=loss(C_1,dists,light_strengths,C_0=C_0,fit_type=fit_type,loss_type=loss_type)
            ls.append(l)
            ks.append(k)
            bs.append(b)
    
    ls=np.array(ls).reshape(len(C_1s),len(C_0s))

    best_indx=np.argmin(ls)
    best_l=np.min(ls)
    x,y=np.unravel_index(best_indx,ls.shape)
    
    return C_1s[x],C_0s[y],ks[best_indx],bs[best_indx]


## Multi-lateration Localization Algorithm. The shape of readings is (t*num_sensors,), the shape of sensor locs is (t*num_sensors,2). 
## For the algorithm to work, sensor_locs shall be not repetitive, and t*num_sensors shall be >=3.
def rhat(scalar_readings,C1,C0,k,b):
    ## h(r)=k(r-C_1)**b+C_0
    return ((scalar_readings-C0)/k)**(1/b)+C1

def multi_lateration_from_rhat(rhat,sensor_locs):
    
    A=2*(sensor_locs[-1,:]-sensor_locs[:-1,:])
    
    rfront=rhat[:-1]**2
    
    rback=rhat[-1]**2
    
    pback=np.sum(sensor_locs[-1,:]**2)
    
    pfront=np.sum(sensor_locs[:-1,:]**2,axis=1)
    
    B=rfront-rback+pback-pfront

    qhat=np.linalg.pinv(A).dot(B)

    return qhat

def multi_lateration(scalar_readings,sensor_locs,C1=0.07,C0=1.29,k=15.78,b=-2.16):
    rhat=rhat(scalar_readings,sensor_locs,C1,C0,k,b)
    return multi_lateration_from_rhat(rhat,sensor_locs)

## Simple pose to (pose.x,pose.z) utility.
def pose2xz(pose):
    return np.array([pose.position.x,pose.position.z])

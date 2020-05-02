import numpy as np
from sklearn.linear_model import LinearRegression
from geometry_msgs.msg import Pose,Twist
from turtlesim.msg import Pose as tPose
from nav_msgs.msg import Odometry
def stop_twist():
    """
    An all-zero twist.
    """
    vel_msg=Twist()

    # Linear velocity in the x-axis.

    vel_msg.linear.x = 0
    vel_msg.linear.y = 0
    vel_msg.linear.z = 0

    # Angular velocity in the z-axis.
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0

    return vel_msg

def prompt_pose_type_string():
    platform_2_pose_types=dict()
    platform_2_pose_types['s']="turtlesimPose"
    platform_2_pose_types['g']='Odom'
    platform_2_pose_types['t']='Pose'
    platform_2_pose_types['o']='optitrack'

    platform=input("Please indicate the platform of your experiment.\n s => turtlesim\n g => Gazebo\n t => Real Robots Turtlebot3 \n o => Optitrack:")
    return platform_2_pose_types[platform]

def toyaw(pose):
    if type(pose) is tPose:
        return tPose2yaw(pose)
    elif type(pose) is Odometry:
        return yaw_from_odom(pose)
    else:
        print('Pose to yaw conversion is not yet handled for {}'.format(type(pose)))
        return None

def toxy(pose):
    if type(pose) is tPose:
        return tPose2xy(pose)
    elif type(pose) is Odometry:
        return xy_from_odom(pose)
    elif type(pose) is Pose:
        return pose2xz(pose)
    else:
        print('Pose to xy conversion is not yet handled for {}'.format(type(pose)))
        return None

def get_pose_type_and_topic(pose_type,robot_name):
    
    if pose_type=='turtlesimPose':
        pose_type=tPose
        rpose_topic="/{}/pose".format(robot_name)
    elif pose_type=='Pose':
        pose_type=Pose
        rpose_topic="/{}/pose".format(robot_name)
    elif pose_type=='Odom':
        pose_type=Odometry
        rpose_topic="/{}/odom".format(robot_name)
    elif pose_type=='optitrack':
        pose_type=PoseStamped
        rpose_topic="/vrpn_client_node/{}/pose".format(robot_name)

    return pose_type,rpose_topic


"""
pose is the Standard pose type as defined in geometry_msgs
"""
def pose2xz(pose):
    return np.array([pose.position.x,pose.position.z])

"""
tPose stands for the ROS data type turtlesim/Pose
"""
def tPose2xy(data):
        return np.array([data.x,data.y])
def tPose2yaw(data):
        return data.theta
    

"""
The following are the location/yaw converters from Odometry.
"""

def quaternion2yaw(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny_cosp, cosy_cosp)

def yaw_from_odom(odom):
    return quaternion2yaw(odom.pose.pose.orientation)
def xy_from_odom(odom):
    return np.array([odom.pose.pose.position.x,odom.pose.pose.position.y])


def top_n_mean(readings,n):
    """
        top_n_mean is used to convert the reading vector of the 8 light-sensors installed 
        on Turtlebots into a single scalar value, representing the overall influence of 
        the light source to the Turtlebots.
    """

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


    print('fit_type:',fit_type)
    if fit_type=="light_readings":
        ## h(r)=k(r-C_1)**b+C_0
        yhat=k*(dists-C_1)**b+C_0
        
        if loss_type=='max':
            e=np.sqrt(np.max((yhat-light_strengths)**2))
        else:
            e=np.sqrt(np.mean((yhat-light_strengths)**2))
    elif fit_type=='dists':
        rh=rhat(light_strengths,C_1,C_0,k,b)
        if loss_type=='max':
            e=np.sqrt(np.max((rh-dists)**2))
        else:
            e=np.sqrt(np.mean((rh-dists)**2))
    
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



"""
 Multi-lateration Localization Algorithm. The shape of readings is (t*num_sensors,), the shape of sensor locs is (t*num_sensors,2). 
 For the algorithm to work, sensor_locs shall be not repetitive, and t*num_sensors shall be >=3.
"""

def multi_lateration_from_rhat(sensor_locs,rhat):
    
    A=2*(sensor_locs[-1,:]-sensor_locs[:-1,:])
    
    rfront=rhat[:-1]**2
    
    rback=rhat[-1]**2
    
    pback=np.sum(sensor_locs[-1,:]**2)
    
    pfront=np.sum(sensor_locs[:-1,:]**2,axis=1)
    
    B=rfront-rback+pback-pfront

    qhat=np.linalg.pinv(A).dot(B)

    return qhat


def rhat(scalar_readings,C1,C0,k,b):
    ## h(r)=k(r-C_1)**b+C_0
    return ((scalar_readings-C0)/k)**(1/b)+C1

def multi_lateration(sensor_locs,scalar_readings,C1=0.07,C0=1.29,k=15.78,b=-2.16):
    rhat=rhat(scalar_readings,C1,C0,k,b)
    return multi_lateration_from_rhat(sensor_locs,rhat)



"""
    Localization with by initial guess and looking at the closest point of intersections.
"""
def intersection_localization(ps,rs,qhint):
    '''
    ps: shape (n,2)
    rs: shape (n,)
    qhint: shape (2,)
    
    Generate interceptions of circles from ps, rs. Then return the closest interception to qhint.
        
    '''
    intercepts=get_all_intersections(ps,rs)
    if intercepts is None:
        return None
    
    est_loc,_=closest_points(intercepts,qhint.reshape(1,2))
    return est_loc

## This intersection solver provides the building block of an alternative localization solution.
'''
    https://stackoverflow.com/questions/55816902/finding-the-intersection-of-two-circles
'''
def get_intersections(p0, r0, p1, r1):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1
    (x0, y0)=p0
    (x1, y1)=p1
    d=np.sqrt((x1-x0)**2 + (y1-y0)**2)

    # non intersecting
    if d > r0 + r1 :
        return None
    # One circle within other
    if d < abs(r0-r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a=(r0**2-r1**2+d**2)/(2*d)
        h=np.sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d   
        y2=y0+a*(y1-y0)/d   
        x3=x2+h*(y1-y0)/d     
        y3=y2-h*(x1-x0)/d 

        x4=x2-h*(y1-y0)/d
        y4=y2+h*(x1-x0)/d

        return np.array([[x3, y3],[x4, y4]])
def get_all_intersections(ps,rs):
    '''
    ps: an np array with shape (n,2)
    rs: an np array with shape (n,)
    Return value: an np array of coordinates of interceptions, of shape at most (n(n-1),2), 
    since there will be non-intercepting points.
    '''
    rs=np.array(rs)
    ps=np.array(ps)
    n=len(rs)
    assert(ps.shape==(n,2))
    intercepts=[]
    for i in range(n):
        for j in range(i+1,n,1):
            intcpt=get_intersections(ps[i,:],rs[i],ps[j,:],rs[j])
            if not intcpt is None:
                intercepts.append(intcpt)
    if len(intercepts)>0:
        return np.vstack(intercepts)
    else:
        return None
    

def closest_points(qs,ps):
    '''
    qs: shape (m,2)
    ps: shape (n,2)
    Return the closest two points from two sets of points, one from group qs and another from group ps.
    '''
    m=qs.shape[0]
    n=ps.shape[0]
    dists=np.zeros((m,n))

    for i in range(m):
        for j in range(n):
            dists[i][j]=np.linalg.norm(qs[i,:]-ps[j,:])
    q,p=np.unravel_index(np.argmin(dists),shape=(m,n))
    return qs[q],ps[p]


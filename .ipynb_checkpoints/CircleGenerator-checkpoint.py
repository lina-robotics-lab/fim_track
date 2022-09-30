import numpy as np
class CircleGenerator(object):
    """docstring for StraightLineGenerator"""
    def __init__(self,origin,radius,dtheta,n_timesteps = 10):
        super(CircleGenerator, self).__init__()
        self.origin = origin
        self.radius = radius
        self.dtheta = dtheta
        self.n_timesteps = n_timesteps
    def get_waypoints(self,qs):
        qs = qs.flatten()
        d = qs-self.origin
        theta = np.arctan2(d[1],d[0])
        wp = self.origin + \
                self.radius * np.array([[np.cos(theta + self.dtheta*i), np.sin(theta+self.dtheta * i)] \
                                        for i in range(self.n_timesteps)])
        return wp
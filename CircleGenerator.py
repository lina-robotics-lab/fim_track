import numpy as np
class CircleGenerator(object):
    """docstring for CircleGenerator"""
    def __init__(self,origin,radius,dtheta):
        super(CircleGenerator, self).__init__()
        self.origin = origin
        self.radius = radius
        self.dtheta = dtheta
    def next_waypoint(self,qs):
        qs = qs.flatten()
        d = qs-self.origin
        theta = np.arctan2(d[1],d[0])
        wp = self.origin + \
                self.radius * np.array([np.cos(theta + self.dtheta), np.sin(theta+self.dtheta)])
        return wp
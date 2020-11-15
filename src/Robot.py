import numpy as np
class Robot(object):
    """The data container class Robot. """
    def __init__(self, init_loc,name='robot'):
        super(Robot, self).__init__()
        self._loc = np.array(init_loc)
        self._vel = np.zeros(self._loc.shape)
        self._loc_log = []
        self.name=name
    @property
    def loc(self):
        return self._loc
    @property
    def vel(self):
        return self._vel

    @property
    def loc_log(self):
        return self._loc_log
    
    def update_vel(self,next_vel):
        self._vel=next_vel
        return self.vel
    # Record current location to log and update current location to next_loc
    def update_loc(self,next_loc):
        self._loc_log.append(self.loc)
        self._loc = np.array(next_loc)
        return self.loc


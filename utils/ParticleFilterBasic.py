import numpy as np
import numpy.random as random
import scipy.stats as stats

class ParticleFilterBasic:
    '''
    Parameters: dim_x - number of dimensions on the target - should be 4 w/ x, y, xdot, ydot
    dim_z - number of dimensions on sensors - should be # of sensors, each returns scalar
    N - number of particles

    Attributes (assume one target)
    x - state estimate vector - dim_x
    x_post - posterior estimate - dim_x
    x_prior - prior estimate - dim_x


    N - # of particles - scalar
    sensor_std - sensor error for the resampling step - tune? - dim_z
    movement_std - particle propagation error, for exploration - tune? - dim_x

    particles - particles for simulating estimated location - (N, dim_x)
    weights - weights for each particle - (N)

    NOTE: assume that the dt is handled for you in upper layers, update func uses velocity
    w/ dt = 1

    '''

    def __init__(self, dim_x, dim_z, sensor_std=0.01, move_std=0.1, N = 50):

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.N = N

        self.x = np.zeros(dim_x)
        self.x_prior = np.zeros(dim_x)
        self.x_post = np.zeros(dim_x)

        self.sensor_std = sensor_std
        self.move_std = move_std

        self.particles = np.zeros([N, dim_x])
        self.weights = (1 / N) * np.ones(N)


    def init_gaussian(self, mean, std):
        '''
        Initialize particles around a starting point w/ standard deviation in gaussian pattern
        mean - mean area to initialize particles around - dim_x
        std - standard deviation to generate particles in - dim_x
        '''
        self.particles = mean + np.random.randn(self.N, self.dim_x) * std
        self.x = np.average(self.particles, axis=0, weights=self.weights)

    def update(self, new_meas,new_ps, meas_func):
        '''
        Take in measurements, weight and resample particles - update x_post - update velocities
        '''
        # assume x_prior has the raw no measurement average of particles
        
        # weight - location update
        # import pdb
        # pdb.set_trace()
        # for i in range(self.N):
        import pdb
        
        meas_predict = np.apply_along_axis(meas_func,1,self.particles)
        pdf = lambda predict:stats.multivariate_normal(predict,self.sensor_std).pdf(new_meas)
        self.weights = np.apply_along_axis(pdf,1,meas_predict)

        self.weights += 1.e-300      # avoid round-off to zero
        self.weights /= sum(self.weights) # normalize

        # weight - velocity update
        self.x = np.average(self.particles, axis=0, weights=self.weights)
        ref_vel = self.x[0:int(self.dim_x / 2):] - self.x_post[0:int(self.dim_x / 2):]

        # resampling step
        indexs = random.choice(len(self.particles), size=len(self.particles), replace = True, p=self.weights)
        self.particles[:] = self.particles[indexs]
        self.weights.resize(self.N)
        self.weights.fill(1.0 / len(self.weights))

        self.particles[:, int(self.dim_x / 2):] = 0.5 * (ref_vel - self.x[int(self.dim_x / 2):])

        self.x_post = self.x.copy()


    def predict(self):
        '''
        Propagate the particles forward - the estimate of this is the prior
        '''

        self.particles[:, 0:int(self.dim_x / 2)] += self.particles[:, int(self.dim_x / 2):] + np.random.randn(self.N, int(self.dim_x / 2)) * self.move_std
        self.x = np.average(self.particles, axis=0, weights=self.weights)

        self.x_prior = self.x.copy()


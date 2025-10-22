import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

'''
0 < t < 10
v = [1, 0] (vx = 1, vy = 0)

10 < t < 20
v = [0, -1] (vx = 0, vy = -1)

20 < t < 30
v = [-1, 0] (vx = -1, vy = 0)

30 < t < 40
v = [0, 1] (vx = 0, vy = 1)
'''

class ExtendedKalmanFilter:
    def __init__(self, t, t_total, r, q, land_one, land_two):
        self.time = t_total # the total time, 40 sec
        self.dt = t # the timestep, 0.5 sec
        
        # Covs for the process & meas model
        self.r_cov = r
        self.q_cov = q

        # Landmarks 
        self.l1 = land_one
        self.l2 = land_two
    
    def velo(self, t):
        v = None
        if 0 <= t < 10:
            v = np.array([1, 0])
        elif 10 <= t < 20:
            v = np.array([0, -1])
        elif 20 <= t < 30:
            v = np.array([-1, 0])
        elif 30 <= t < 40:
            v = np.array([0, 1])
        return v

    # State transition model
    def g(self, mu_prev):
        px, py, vx, vy = mu_prev
        px_next = px + self.dt * vx
        py_next = py + self.dt * vy

        new_arr = np.array([px_next, py_next, vx, vy])
        return new_arr
    
    def g_jacobian(self):
        """
        A constant jacobian of the state estimation model g().
        This is a 4x4 matrix for each of the 4 state vars px, py, vx, and vy,
        and the states [px + dt * vx, py + dt * vy, vx, vy].
        """
        jacobian = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        return jacobian

    def h(self, mu):
        px, py, vx, vy = mu
        dist_one = np.sqrt((self.l1[0] - px)**2 + (self.l1[1] - py)**2)
        dist_two = np.sqrt((self.l2[0] - px)**2 + (self.l2[1] - py)**2)
        z_t = np.array([
            dist_one,
            dist_two
        ])
        return z_t

    def h_jacobian(self, mu):
        """
        Jacobian math for the measurement model h().
        This is a 2x4 matrix, with 1x2 input matrix for the h() euclidean distance
        between the believed pose and two landmarks. There are 4 state vars, for a 2x4 matrix.
        """
        px, py, vx, vy = mu
        dist_one = np.sqrt((self.l1[0] - px)**2 + (self.l1[1] - py)**2)
        dist_two = np.sqrt((self.l2[0] - px)**2 + (self.l2[1] - py)**2)

        ind_zero = (self.l1[0] - px) / dist_one
        ind_one = (self.l1[1] - py) / dist_one
        ind_two = (self.l2[0] - px) / dist_two
        ind_three = (self.l2[1] - py) / dist_two
        
        jacobian = np.array([
            [ind_zero, ind_one, 0, 0],
            [ind_two, ind_three, 0, 0]
            ])
        return jacobian

    def EKF(self, mean_init, cov_init, z_t):
        mean_hat = self.g(mean_init)

        g_jacob = self.g_jacobian()
        h_jacob = self.h_jacobian(mean_hat)

        cov_hat = g_jacob @ cov_init @ g_jacob.T + self.r_cov
        print(cov_hat)

        k_gain = cov_hat @ h_jacob.T @ np.linalg.inv(h_jacob @ cov_hat @ h_jacob.T + self.q_cov)

        mean = mean_hat + k_gain @ (z_t - self.h(mean_hat))
        print(mean)
        cov = (np.eye(4) - k_gain @ h_jacob) @ cov_hat
        print(cov)
        return mean, cov
    
    def process(self, mean_init, cov_init, z_t):

        t = 0
        while t < self.time:

            # Get the velocity array
            v = self.velo(t)
            mean_init, cov_init = self.EKF(mean_init, cov_init, z_t)

            # Update the time
            t += self.dt
        
dt = 0.5
time_total = 40

#r = 0.1 * np.eye(2)
r = np.diag([0.1, 0.1, 0.0, 0.0])

q = 0.5 * np.eye(2)

l1 = np.array([5, 5])
l2 = np.array([-5, 5])

#Prior / Initial Pose Belief
x = np.array([0, 0, 0, 0])  # belief about mean
p = np.eye(4) #belief about cov

z_true = np.array([
    np.linalg.norm(l1 - np.array([0, 0])),
    np.linalg.norm(l2 - np.array([0, 0]))
])

noise = np.random.multivariate_normal(mean=[0, 0], cov=q)
z_t = z_true + noise

ekf = ExtendedKalmanFilter(dt, time_total, r, q, l1, l2)
print(ekf.process(x, p, z_t))

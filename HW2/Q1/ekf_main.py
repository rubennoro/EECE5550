import numpy as np
import matplotlib.pyplot as plt

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

        self.velocity = [0, 0]
    
    def velo(self, t):
        if 0 <= t <= 10:
            self.velocity = np.array([1, 0])
        elif 10 < t <= 20:
            self.velocity = np.array([0, -1])
        elif 20 < t <= 30:
            self.velocity = np.array([-1, 0])
        elif 30 < t < 40:
            self.velocity = np.array([0, 1])
        
    # State transition model
    def state_transition(self, mu_prev):
        vx, vy = self.velocity
        px, py = mu_prev
        px_next = px + self.dt * vx
        py_next = py + self.dt * vy

        return np.array([px_next, py_next])
    
    def state_jacobian(self):
        """
        A constant jacobian of the state estimation model g().
        This is a 4x4 matrix for each of the 2 state vars px, py.
        and the states [px + dt * vx, py + dt * vy].
        """
        return np.array([
            [1, 0],
            [0, 1]
        ])

    def measurement_model(self, mu):
        px, py = mu
        dist_one = np.sqrt((self.l1[0] - px)**2 + (self.l1[1] - py)**2)
        dist_two = np.sqrt((self.l2[0] - px)**2 + (self.l2[1] - py)**2)
        return np.array([dist_one, dist_two])

    def meas_jacobian(self, mu):
        """
        Jacobian math for the measurement model h().
        This is a 2x4 matrix, with 1x2 input matrix for the h() euclidean distance
        between the believed pose and two landmarks. There are 4 state vars, for a 2x4 matrix.
        """
        #vx, vy = self.velocity
        px, py = mu
        dist_one = np.sqrt((self.l1[0] - px)**2 + (self.l1[1] - py)**2)
        dist_two = np.sqrt((self.l2[0] - px)**2 + (self.l2[1] - py)**2)
        
        h00 = (px - self.l1[0]) / dist_one
        h01 = (py - self.l1[1]) / dist_one
        
        h10 = (px - self.l2[0]) / dist_two
        h11 = (py - self.l2[1]) / dist_two
        
        jacobian = np.array([
            [h00, h01],
            [h10, h11]
            ])
        return jacobian

    def EKF(self, mean_init, cov_init, z_t):
        # This outputs a 2x1 vector for px, py
        mean_hat = self.state_transition(mean_init)
        #print(mean_hat)
        g_jacob = self.state_jacobian()
        cov_hat = g_jacob @ cov_init @ g_jacob.T + self.r_cov

        # 2x2 jacobian for the meas model
        h_jacob = self.meas_jacobian(mean_hat)
        z_pred = self.measurement_model(mean_hat)

        S = h_jacob @ cov_hat @ h_jacob.T + self.q_cov
        k_gain = cov_hat @ h_jacob.T @ np.linalg.inv(S)

        y = z_t - z_pred
        #print(z_pred)
        mean = mean_hat + k_gain @ y
        cov = (np.eye(2) - k_gain @ h_jacob) @ cov_hat
        #print(mean)
        return mean, cov
    
    def process(self, mean_init, cov_init, z_t):
        """
        mean_init starts as [0, 0]
        cov_init starts as I_2
        z_t starts as [0, 0]
        """
        t = 0
        true_positions = []
        estimated_means = []
        estimated_covs = []
        true_pos = mean_init
        mean_pred = mean_init 

        while t < self.time:
            self.velo(t)

            # True position starts at 0, 0 and then moves forward after t > 0
            true_pos = self.state_transition(true_pos)
            true_positions.append(true_pos)

            z_t = self.measurement_model(true_pos) + np.random.multivariate_normal(mean=[0, 0], cov=self.q_cov)
            #print(new_meas)
            
            mean_pred, cov_init = self.EKF(mean_pred, cov_init, z_t)
            
            estimated_means.append(mean_pred)
            estimated_covs.append(cov_init)
            # Update the time
            t += self.dt
        return np.array(true_positions), np.array(estimated_means), np.array(estimated_covs)
# Time params 
dt = 0.5
time_total = 40

# The Cov for the state model.
R = 0.1 * np.eye(2)

# The Cov for the measurement model.
Q = 0.5 * np.eye(2)

# Initial landmarks
l1 = np.array([5, 5])
l2 = np.array([-5, 5])

#Prior / Initial Pose Belief
x = np.array([0, 0])  # belief about pose
p = np.eye(2) #belief about cov

z_t = np.array([0, 0])

ekf = ExtendedKalmanFilter(dt, time_total, R, Q, l1, l2)

# Get the true robot positions, estimated means, and estimated covariances
true_positions, estimated_means, estimated_covs = ekf.process(x, p, z_t)
sigma_bounds = np.sqrt(np.array([np.diag(P) for P in estimated_covs]))[:, :2] * 3

plt.figure(figsize=(15, 15))  # Adjusted figure size
plt.scatter([l1[0], l2[0]], [l1[1], l2[1]], c='black', marker='.', label='Landmarks')
plt.plot(true_positions[:,0], true_positions[:,1], 'b-', label='True Path')
plt.plot(estimated_means[:, 0], estimated_means[:, 1], 'g-', label='Estimated Path')

plt.xlabel('x pos')
plt.ylabel('y pos')
plt.title('actual path vs. estimated path')
plt.legend()
plt.grid(True)

plt.figure(figsize=(10, 5))  # Adjusted figure size
timesteps = time_total / dt
plt.plot(range(int(timesteps)), sigma_bounds[:, 0], 'rp-', label='x 3 sigma')
plt.plot(range(int(timesteps)), sigma_bounds[:, 1], 'yp-', label='y 3 sigma')

plt.xlabel('time step')
plt.ylabel('3 sigma')
plt.title('3 sigma plotted against time')
plt.legend()
plt.grid(True)

plt.show()
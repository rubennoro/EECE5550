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

        self.velocity = [0, 0]
    
    def velo(self, t):
        if 0 <= t < 10:
            self.velocity = np.array([1, 0])
        elif 10 <= t < 20:
            self.velocity = np.array([0, -1])
        elif 20 <= t < 30:
            self.velocity = np.array([-1, 0])
        elif 30 <= t < 40:
            self.velocity = np.array([0, 1])
        
    # State transition model
    def g(self, mu_prev):
        vx, vy = self.velocity
        px, py = mu_prev
        px_next = px + self.dt * vx
        py_next = py + self.dt * vy

        new_arr = np.array([px_next, py_next])
        return new_arr
    
    def g_jacobian(self):
        """
        A constant jacobian of the state estimation model g().
        This is a 4x4 matrix for each of the 2 state vars px, py.
        and the states [px + dt * vx, py + dt * vy].
        """
        jacobian = np.array([
            [1, 0],
            [0, 1]
        ])
        return jacobian

    def h(self, mu):
        px, py = mu
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
        vx, vy = self.velocity
        px, py = mu
        dist_one = np.sqrt((self.l1[0] - px)**2 + (self.l1[1] - py)**2)
        dist_two = np.sqrt((self.l2[0] - px)**2 + (self.l2[1] - py)**2)

        ind_zero = (self.l1[0] - px) / dist_one
        ind_one = (self.l1[1] - py) / dist_one
        ind_two = (self.l2[0] - px) / dist_two
        ind_three = (self.l2[1] - py) / dist_two
        
        jacobian = np.array([
            [ind_zero, ind_one],
            [ind_two, ind_three]
            ])
        return jacobian

    def EKF(self, mean_init, cov_init, z_t):
        # This outputs a 2x1 vector for px, py
        mean_hat = self.g(mean_init)

        # 2x2 jacobian for the state propagation
        g_jacob = self.g_jacobian()

        # 2x2 jacobian for the meas model
        h_jacob = self.h_jacobian(mean_hat)

        cov_hat = g_jacob @ cov_init @ g_jacob.T + self.r_cov
        print(cov_hat)

        k_gain = cov_hat @ h_jacob.T @ np.linalg.inv(h_jacob @ cov_hat @ h_jacob.T + self.q_cov)

        mean = mean_hat + k_gain @ (z_t - self.h(mean_hat))
        print(mean)
        cov = (np.eye(2) - k_gain @ h_jacob) @ cov_hat
        print(cov)
        return mean, cov
    
    def process(self, mean_init, cov_init, z_t):

        t = 0
        true_positions = []
        estimated_means = []
        estimated_covs = []

        while t < self.time:

            true_positions.append(self.g(mean_init))
            # Update the velocity
            self.velo(t)

            mean_init, cov_init = self.EKF(mean_init, cov_init, z_t)

            estimated_means.append(mean_init)
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

z_true = np.array([
    np.linalg.norm(l1 - np.array([0, 0])),
    np.linalg.norm(l2 - np.array([0, 0]))
])

noise = np.random.multivariate_normal(mean=[0, 0], cov=Q)
z_t = z_true + noise

ekf = ExtendedKalmanFilter(dt, time_total, R, Q, l1, l2)

'''
GPT code below.
'''
# Get the true robot positions, estimated means, and estimated covariances
true_positions, estimated_means, estimated_covs = ekf.process(x, p, z_t)

# Plotting
plt.figure(figsize=(10, 8))

# Plot landmarks
plt.scatter(l1[0], l1[1], color='r', label="Landmark 1", zorder=5)
plt.scatter(l2[0], l2[1], color='g', label="Landmark 2", zorder=5)

# Plot true robot trajectory
true_positions = np.array(true_positions)
plt.plot(true_positions[:, 0], true_positions[:, 1], label="True Path", color='blue', linewidth=2)

# Plot estimated robot trajectory with 3-sigma confidence bounds
estimated_means = np.array(estimated_means)
for i in range(0, len(estimated_means), 5):  # Show every 5th point for clarity
    mean = estimated_means[i]
    cov = estimated_covs[i]

    # Extract the standard deviations (3σ bounds)
    std_dev = np.sqrt(np.diag(cov))  # Standard deviations in x and y
    # Check if the standard deviations are large enough
    if std_dev[0] > 0.1 and std_dev[1] > 0.1:  # Threshold to avoid plotting very small ellipses
        # Plot the ellipse with 3σ bounds
        ellipse = Ellipse(mean, 3 * std_dev[0], 3 * std_dev[1], edgecolor='orange', facecolor='none', linewidth=1)
        plt.gca().add_patch(ellipse)
        
# Plot estimated mean path
plt.plot(estimated_means[:, 0], estimated_means[:, 1], label="Estimated Path (Mean)", color='orange', linestyle='--', linewidth=2)

plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("EKF Robot Localization with 3σ Confidence Bounds")
plt.legend(loc='best')
plt.grid(True)
plt.show()

"""
State Estimation by Particle Filtering on a Lie Group.
"""
import numpy as np
import matplotlib.pyplot as plt

"""
Matrix representation for pose in SE(2).
"""
class Pose:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
    
    def matrix(self):
        return np.array([
            [np.cos(self.theta), -np.sin(self.theta), self.x],
            [np.sin(self.theta), np.cos(self.theta), self.y],
            [0, 0, 1]
        ])
    
"""
Particle Filter Object.
"""
class ParticleFilter:
    def __init__(self, wheel_l_cov, wheel_r_cov, radius, width, left_speed, right_speed):
        self.l_cov = wheel_l_cov
        self.r_cov = wheel_r_cov
        self.radius = radius
        self.track_width = width
        self.left_speed = left_speed
        self.right_speed = right_speed

    def wheel_speed_l(self, speed_l):
        """
        Get the new left wheel speed based on the new command.
        """
        new_speed = speed_l + np.random.normal(0, self.l_cov)
        return new_speed

    def wheel_speed_r(self, speed_r):
        """
        Get the new right wheel speed based on the new command.
        """
        new_speed = speed_r + np.random.normal(0, self.r_cov)
        return new_speed
    
    """
    Part C.
    """
    def propagation(self, X, t1, t2):
        """
        Input: Particle set sampled from belief p(xt) over initial state and control.
        Output: Particle set sampled from the belief p(x_t+1|u_t) over the subsequent state after control.
        """
        dt = t2 - t1
        # Init empty particle set
        n = len(X)
        samples = []
        for particle in X:
            theta_t = np.arctan2(particle[1, 0], particle[0, 0])
            # x_t = particle[0, 2]
            # y_t = particle[1, 2]
            
            # Sample the wheel .
            left_speed = self.wheel_speed_l(self.left_speed)
            right_speed = self.wheel_speed_r(self.right_speed)

            # Update velocity and angular frequency.
            velo = (self.radius / 2) * (left_speed + right_speed)
            angular_freq = (self.radius / self.track_width) * (right_speed - left_speed)

            # Get complete pose update.
            updated_theta = angular_freq * dt
            updated_x = (velo / angular_freq) * (np.sin(theta_t + updated_theta) - np.sin(theta_t))
            updated_y = -(velo / angular_freq) * (np.cos(theta_t + updated_theta) - np.cos(theta_t))
            
            delta_T = np.array([
                [np.cos(updated_theta), -np.sin(updated_theta), updated_x],
                [np.sin(updated_theta), np.cos(updated_theta), updated_y],
                [0, 0, 1]
            ])
            new_sample = particle @ delta_T
            samples.append(new_sample)
        return samples

    """
    Part D.
    """
    def update(self, X, measurement, noise_mag):
        """
        Params:
        X: particle set of prior pose belief
        measurement: Noisy position z_t sampled through generative model after state propagation.
        noise_mag: magnitude of measurement noise

        1) Calculates importance weight for each particle with MLE.
        2) Initialize empty particle set
        3) Importance-weighted resampling with replacement. 
        Sample probabilities proportional to importanec weights.
        Adds sample to the posterior particle set.
        """
        weights = []
        # Step 1
        for particle in X:
            pose = particle[0 : 2, 2]

            diff = np.linalg.norm(measurement - pose)

            weight = np.exp(-(diff**2) / (2 * noise_mag**2))
            weights.append(weight)

        weights = np.array(weights)
        weights /= np.sum(weights)
        new_particles = []

        rand = np.random.uniform(0, 1 / len(X))
        c = weights[0]
        j = 0
        for i in range(len(X)):
            threshold = rand + i / len(X)
            while threshold > c:
                j += 1
                c += weights[j]
            new_particles.append(X[i])
        
        new_particles = np.array(new_particles)
        return new_particles
"""
Given Parameters.
"""
l_wheel_cov = 0.05
r_wheel_cov = 0.05

p_cov = 0.10

rad = 0.25
width = 0.5

left_wheel_init = 1.5
right_wheel_init = 2

"""
Object Instantiation
"""
particle_filter = ParticleFilter(l_wheel_cov, r_wheel_cov, rad, width, left_wheel_init, right_wheel_init)

"""
Part E.
"""
N = 1000
t_0 = 0
t_1 = 10
init_particles = [Pose(0, 0, 0).matrix() for _ in range(N)]
particle_set = particle_filter.propagation(init_particles, t_0, t_1)

# Collect (x,y) Poses
poses = np.array([(pose[0, 2], pose[1, 2]) for pose in particle_set])

# Mean and Cov 
mean_pos = np.mean(poses, axis=0)
covariance_pos = np.cov(poses, rowvar=False)

print("Mean: ", mean_pos)
print("Cov: \n", covariance_pos)

plt.figure(figsize=(8, 6))
plt.scatter(poses[:, 0], poses[:, 1], s=10, c='purple', label='particles')
plt.scatter(mean_pos[0], mean_pos[1], c='green', marker='o', label='mean')
plt.xlabel('X Pos')
plt.ylabel('Y Pos')
plt.title('Particles for t=10')
plt.legend()
plt.grid(True)

plt.show()

"""
Part F.
"""
# Reinstantiate object 

particle_filter = ParticleFilter(l_wheel_cov, r_wheel_cov, rad, width, left_wheel_init, right_wheel_init)
# Define time intervals
times = [0, 5, 10, 15, 20]

# Initialize
particle_sets = []
current_particles = [Pose(0, 0, 0).matrix() for _ in range(N)]

for i in range(1, len(times)):
    t1, t2 = times[i - 1], times[i]
    
    # Propagate from t1 to t2
    current_particles = particle_filter.propagation(current_particles, t1, t2)
    
    # Store the result for plotting later
    particle_sets.append(current_particles)
    
    # Extract (x, y) positions
    poses = np.array([(pose[0, 2], pose[1, 2]) for pose in current_particles])
    
    # Compute mean and covariance
    mean_pos = np.mean(poses, axis=0)
    covariance_pos = np.cov(poses, rowvar=False)
    
    print(f"Mean {t1} to {t2}: {mean_pos}")
    print(f"Cov {t1} to {t2}:\n{covariance_pos}\n")

titles = [f't = {t}' for t in times[1:]]
colors = ['purple', 'blue', 'orange', 'green']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, particles, title, color in zip(axes, particle_sets, titles, colors):
    positions = np.array([(p[0, 2], p[1, 2]) for p in particles])
    ax.scatter(positions[:, 0], positions[:, 1], s=10, c=color, alpha=0.6)
    
    # Mean position
    mean_pos = np.mean(positions, axis=0)
    ax.scatter(mean_pos[0], mean_pos[1], c='green', marker='o', s=50, label='mean')
    
    ax.set_title(title)
    ax.set_xlabel('x position')
    ax.set_ylabel('y position')
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

"""
Part G.
"""
N = 1000
z_5 = np.array([1.6561, 1.2847])
z_10 = np.array([1.0505, 3.1059])
z_15 = np.array([-0.9875, 3.2118])
z_20 = np.array([-1.6450, 1.1978])
z_arr = np.array([z_5, z_10, z_15, z_20])

times = [0, 5, 10, 15, 20]
particle_sets = []
current_particles = [Pose(0, 0, 0).matrix() for _ in range(N)]

avg_poses = []

for i in range(1, len(times)):
    t1, t2 = times[i - 1], times[i]

    current_particles = particle_filter.propagation(current_particles, t1, t2)

    meas = z_arr[i - 1]

    # p_cov is the meas noise
    current_particles = particle_filter.update(current_particles, meas, p_cov)

    poses = np.array([(pose[0, 2], pose[1, 2]) for pose in current_particles])

    mean_pos = np.mean(poses, axis=0)
    avg_poses.append(mean_pos)
    plt.scatter(poses[:, 0], poses[:, 1], s=10, c=colors[i-1], label=f't = {times[i]}')

avg_poses = np.array(avg_poses)
sq_errs = np.sum(np.diff(avg_poses, axis=0)**2, axis=1)

print("Error between timesteps:")
for i, err in enumerate(sq_errs):
    print(f"{times[i]} to {times[i+1]}: {err:.4f}")
plt.xlabel('X pos')
plt.ylabel('Y pos')
plt.title('Particle Filter w/ Meas. Updates')
plt.legend()
plt.grid(True)
plt.show()
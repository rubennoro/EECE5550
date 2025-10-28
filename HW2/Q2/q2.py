print("Q2")
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree

file_path_x = 'pclX.txt'
file_path_y = 'pclY.txt'

X = []
Y = []

'''
Extracts x,y,z coords from PCL X
'''
with open(file_path_x, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
            X.append([x, y, z])

'''
Extracts x,y,z coords from PCL Y
'''
with open(file_path_y, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
            Y.append([x, y, z])
            
class ScanMatching:
    def __init__(self):
        pass

    def RSME(self, X, Y, t_init, R_init, d_max, num_ICP_iters):
        """
        Contains all inputs for the functions below, as RMSE() runs ICP to get 
        optimal translation and rotation outputs, and then calculates rmse.
        """
        t, R, C = self.ICP(X, Y, t_init, R_init, d_max, num_ICP_iters)
        
        K = len(C)
        errs = []
        for i, j in C:
            x_transformed = R @ X[i] + t
            rmses = np.linalg.norm(Y[j] - x_transformed)**2
            errs.append(rmses)
        rmse = np.sqrt(np.mean(errs))
        
        return t, R, rmse
    
    def ICP(self, X, Y, t, R, d_max, iters):
        """
        X and Y are input pointclouds.
        t and R are the initial rigid transformations updated by ComputerOptimalRigidRegistration()
        iters and d_max are params
        """
        for i in range(iters):
            C = self.EstimateCorrespondences(X, Y, t, R, d_max)
            t, R = self.ComputeOptimalRigidRegistration(X, Y, C)

        return t, R, C
    
    def EstimateCorrespondences(self, X, Y, t, R, d_max):
        """
        X and Y are input pointclouds. 
        t and R is the initial rigid transformation estimation.
        d_max is maximum admissible distance for associating two points

        Returns a list C {i, j} of estimated correspondences
        """
        C = [] # .append(i, j) for indices in X and Y that match
        nx = len(X)

        tree = cKDTree(Y)
        trans_x = (R @ X.T).T + t
        dists, inds = tree.query(trans_x, distance_upper_bound=d_max)
        for i in range(nx):
            if dists[i] < d_max:
                C.append([i, inds[i]])

        '''
        Runtime for Code below is too long, optimized with above version.
        # For each point in X, find the closest point in Y after transformation.
        for i in range(nx):
            
            x_transformed = np.dot(R, X[i]) + t

            y_j = [np.linalg.norm(Y[j] - x_transformed)**2 for j in range(len(Y))]
            
            # Minimum Index and Distance respectively.
            j_min = np.argmin(y_j)
            min_dist = y_j[j_min]

            # Remove the square root to compare with d_max
            if np.sqrt(min_dist) < d_max:
                
                # Append the X, Y indices from each PCL 
                C.append([i, j_min])
        '''
        return C
    
    def ComputeOptimalRigidRegistration(self, X, Y, C):
        """
        X, Y are input pointclouds. C is a list of point correspondences.
        1) Calculate pointcloud centroids
        2) Calculate deviations of each point in each pointcloud from their centroid
        3) Cross-covariance matrix W
        4) SVD: W = UEV^T 
        5) Construct optimal rotation 
        6) Recover optimal translation
        
        Returns the updated rigid transformation (t, R)
        """

        # Centroids
        K = len(C)
        X_corr = np.array([X[i] for i, _ in C])  # Shape: (K, 3)
        Y_corr = np.array([Y[j] for _, j in C])  # Shape: (K, 3)

        x_hat = np.mean(X_corr, axis=0)
        y_hat = np.mean(Y_corr, axis=0)
        # For each pair of indices in C, get the values associated with each of them
        # and center them. 
        X_center = X_corr - x_hat
        Y_center = Y_corr - y_hat

        # 3 x 3 cross-cov mat
        cross_cov = (Y_center.T @ X_center) / K
        
        U, s, Vt = np.linalg.svd(cross_cov)
        
        rot = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt.T
        trans = y_hat - rot @ x_hat
        return trans, rot
    
X = np.array(X)
Y = np.array(Y)
t_init = np.array([0, 0, 0])
R_init = np.eye(3)
d_max = 0.25
num_ICP_iters = 30
p2 = ScanMatching()
t, R, rmse = p2.RSME(X, Y, t_init, R_init, d_max, num_ICP_iters)

print("Translation:")
print(t)
print("Rotation")
print(R)
print("Root-Mean Squared Error: ")
print(rmse)

x_transformed = [((R @ X[i]) + t) for i in range(len(X))]

# Plotting both point clouds on the same 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot original Y in blue
Y = np.array(Y)
ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], color='b', label='Point Cloud Y')

# Plot transformed X in red
X_transformed = np.array(x_transformed)
ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2], color='r', label='Transformed Point Cloud X')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Point Cloud Matching: Y (Blue) and Transformed X (Red)')
ax.legend()

plt.show()

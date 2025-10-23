print("Q2")
import numpy as np

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

    def ICP(self, X, Y, t, R, d_max, iters):
        C = None
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
        x_hat = sum(X[k[0]] for k in C) / K
        y_hat = sum(Y[k[1]] for k in C) / K

        # For each pair of indices in C, get the values associated with each of them
        # and center them. 
        X_center = [X[k[0]] - x_hat for k in C]
        Y_center = [Y[k[1]] - y_hat for k in C]

        # 3 x 3 cross-cov mat
        W = np.zeros((len(X[0]), len(Y[0])))
        for k in range(K):
            
            # This is the same as 3x1 * 1x3, where X gets transposed
            W += np.outer(Y_center[k], X_center[k])

        W /= K
        print(W)

        U, s, Vh = np.linalg.svd(W)

        print(s)
        v_reg = np.transpose(Vh)
        det = np.linalg.det(np.dot(U, v_reg))
        D = np.diag(np.concatenate([np.ones(2), [det]]))
        
        R = np.dot(U, np.dot(D, Vh))
        t = y_hat - np.dot(R, x_hat)
        return t, R
    
t_init = np.array([0, 0, 0])
R_init = np.eye(3)
d_max = 0.25
num_ICP_iters = 30
p2 = ScanMatching()
print(p2.ICP(X, Y, t_init, R_init, d_max, num_ICP_iters))

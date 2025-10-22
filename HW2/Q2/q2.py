print("Q2")

class ScanMatching:
    def __init__(self):
        pass

    def EstimateCorrespondences(self, X, Y, t, R, d_max):
        """
        X and Y are input pointclouds. 
        t and R is the initial rigid transformation estimation.
        d_max is maximum admissible distance for associating two points

        Returns a list C {i, j} of estimated correspondences
        """
        C = [] # .append(i, j) for indices in X and Y that match
        # for i in len(X):
            # Find the closest point yj in Y to image of xi under the transformation (t, R):
            # y_j = argmin||y-(Rxi + t||^2
            # if ||y_j - (Rx + t)|| < d_max
                # add it to point correspondences C
            
        return
    
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
        return
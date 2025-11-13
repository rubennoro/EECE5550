'''
Question 2 Code.

Estimating the pose of the camera by solving the nonlinear least-squares problem.
Minimize the sum of squared norms of reprojection errors for the observed points.
Camera Projection function pi.
'''

import cv2
import numpy as np
from apriltag import apriltag
import gtsam

imagepath = 'HW3/Q2/vslam/frame_0.jpg'
image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
detector = apriltag("tag36h11")

# Body-centric coords for the AprilTag0's corners.
tag_len = 0.01 # meters
lb = np.array([-tag_len / 2, tag_len / 2, 0])
rb = np.array([tag_len / 2, tag_len / 2, 0])
rt = np.array([tag_len / 2, -tag_len / 2, 0])
lt = np.array([-tag_len / 2, -tag_len / 2, 0])
tag_corners = [lb, rb, rt, lt]

'''
Knowns:
- AprilTag body-centric coord frame corner points
- AprilTag 2D corner points from Camera Image
- Calibration Matrix K

2d Apriltag corner points = K [R|t] P where [R|t] is the camera pose X

Variables:
Camera Poses X
AprilTag Corner Points
'''

class PnP:
    def __init__(self, detector, image):
        self.detector = detector
        self.image = image
        self.graph = gtsam.NonlinearFactorGraph()
    
    def extract_corners(self):
        '''
        [[703.04492188 541.47503662]
        [882.42858887 534.94506836]
        [875.32348633 358.56542969]
        [696.24658203 362.67459106]]
        '''
        detections = self.detector.detect(self.image)
        for d in detections:
            if d['id'] == 0:
                
                ptLB, ptRB, ptRT, ptLT = d['lb-rb-rt-lt']
                print( np.array([ptLB, ptRB, ptRT, ptLT]))
                return np.array([ptLB, ptRB, ptRT, ptLT])
                
        return [-1, -1, -1, -1]

    def build_graph(self, K, tag_corners):
        '''
        Builds the factor graph with the goal of setting up an optimization equation 
        to minimize projection error, so that the camera pose can be estimated as accurately as possible.
        To minimize error, the camera pose must shift, leading to a lower error in the projection / more accurate pose estimation.
        '''
        corners = self.extract_corners()
        k_matrix = gtsam.Cal3_S2(K[0][0], K[1][1], K[0][1], K[0][2], K[1][2])

        # Set an initial guess for the camera pose (e.g., a slight offset)
        initial_pose = gtsam.Pose3()

        initial_estimates = gtsam.Values()

        pose_symbol = gtsam.symbol('x', 0)
        initial_estimates.insert(pose_symbol, initial_pose)


        object_points = np.array(tag_corners, dtype=np.float32)
        image_points = np.array(corners, dtype=np.float32)
        success, rvec, tvec = cv2.solvePnP(object_points, image_points, K[:3,:3], None, flags=cv2.SOLVEPNP_ITERATIVE)
        if success:
            R, _ = cv2.Rodrigues(rvec)
            # Convert to GTSAM pose (note: OpenCV and GTSAM might have different coordinate conventions)
            initial_pose = gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(tvec.flatten()))
             #print("Using OpenCV solvePnP initial pose")
        
        initial_estimates = gtsam.Values()

        pose_symbol = gtsam.symbol('x', 0)
        initial_estimates.insert(pose_symbol, initial_pose)
        
        # Add AprilTag corner points as known 3D points with constrained priors
        tag_corner_symbols = []
        for i, corner in enumerate(tag_corners):
            point_symbol = gtsam.symbol('p', i)
            tag_corner_symbols.append(point_symbol)
            
            # Create Point3 object for the 3D corner point
            point_3d = gtsam.Point3(*corner)
            initial_estimates.insert(point_symbol, point_3d)
            
            point_prior_noise = gtsam.noiseModel.Constrained.All(3)
            self.graph.add(gtsam.PriorFactorPoint3(
                point_symbol, 
                point_3d, 
                point_prior_noise
            ))
        
        for i in range(4):
            
            meas = gtsam.Point2(corners[i][0], corners[i][1])
            factor = gtsam.GenericProjectionFactorCal3_S2(
                meas,  # 2D corner point
                gtsam.noiseModel.Isotropic.Sigma(2, 1.0),  # noise model
                pose_symbol,  # camera pose variable
                tag_corner_symbols[i],  # 3D corner of the tag
                k_matrix  # Camera intrinsic parameters
            )
            # print(factor)
        
            self.graph.add(factor)
        
        pose_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])  # Larger sigmas for weaker prior
        )
        self.graph.add(gtsam.PriorFactorPose3(pose_symbol, initial_pose, pose_prior_noise))
        
        return initial_estimates
    
    def estimate_pose(self, K, tag_corners):
        
        # Construct the graph attribute
        init_ests = self.build_graph(K, tag_corners)
        #print(init_ests)
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, init_ests)
        result = optimizer.optimize()
        # print(f"Optimizer finished in {optimizer.iterations()} iterations.")
        return result.atPose3(gtsam.symbol('x', 0))


K = np.array([[1482.258658843503, 0, 549.5399528638159],
    [0, 1476.486249356242, 936.7846042022142],
    [0, 0, 1]])
persp_n_point = PnP(detector, image)
print(persp_n_point.estimate_pose(K, tag_corners))


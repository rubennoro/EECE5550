'''
Run this from the root directory.
Question 3 Code:
Jointly estimating the camera pose and apriltag poses for VisualSLAM using a factor graph and MLE.
'''

import gtsam
import numpy as np
import cv2
from apriltag import apriltag
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 500 total images
num_images = 500

class VisualSLAM:
    def __init__(self, num_images):
        self.num_imgs = num_images
        self.tags = []
        self.graph = gtsam.NonlinearFactorGraph()  # GTSAM factor graph
        self.initial_values = gtsam.Values()  # Initial values for optimization
        self.noise_model = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)  # Isotropic noise model

    def collect_tags(self):
        '''
        Collects all the apriltags from all images. The self.tags becomes a list of lists
        with each images' poses collected for their AprilTags.
        '''
        for i in range(self.num_imgs):
            self.path = f'HW3/Q2/vslam/frame_{i}.jpg'
            self.tags.append(self.detect_tags())
        
        print("Collection Complete.\n")
    
    def detect_tags(self):
        '''
        Collects the pose of each apriltag for each image.
        '''
        image = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        detector = apriltag("tag36h11")
        detections = detector.detect(image)

        tags = []
        for tag in detections:
            # print(tag)
            id = tag['id']
            
            # Body-centric coords for the AprilTag0's corners.
            tag_len = 0.01 # meters
            lb = np.array([-tag_len / 2, tag_len / 2, 0])
            rb = np.array([tag_len / 2, tag_len / 2, 0])
            rt = np.array([tag_len / 2, -tag_len / 2, 0])
            lt = np.array([-tag_len / 2, -tag_len / 2, 0])
            tag_3d_points = np.array([lb, rb, rt, lt])

            # Camera intrinsic parameters
            K = np.array([
                [1482.26, 0, 549.54],
                [0, 1476.49, 936.78],
                [0, 0, 1]
            ], dtype=np.float32)

            tag_2d_points = np.array(tag['lb-rb-rt-lt'], dtype=np.float32)
            
            # Use this matrix in functions like solvePnP:
            _, rvec, tvec = cv2.solvePnP(tag_3d_points, tag_2d_points, K[:3, :3], None, flags=cv2.SOLVEPNP_ITERATIVE)
            R, _ = cv2.Rodrigues(rvec)
            pose = gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(tvec.flatten()))
            
            # Pose between Cam and each AprilTag
            tags.append([id, pose])

        return tags

    def construct_graph(self):
        '''
        Building the factor graph.
        '''

        # Prior Factor is AprilTag 0 at world origin
        self.graph.add(
            gtsam.PriorFactorPose3(
                gtsam.symbol('t', 0),
                gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 0, 0)),
                gtsam.noiseModel.Constrained.All(6)
            )
        )
        # Initialize it in initial values
        self.initial_values.insert(gtsam.symbol('t', 0), gtsam.Pose3())

        # For each image
        for i, detections in enumerate(self.tags):
            cam_key = gtsam.symbol('x', i)

            # Add initial guess for camera if not first frame
            if not self.initial_values.exists(cam_key):
                self.initial_values.insert(cam_key, gtsam.Pose3())

            # Get the pose from each AprilTag.
            for tag_id, tag_pose_cam in detections:
                tag_key = gtsam.symbol('t', tag_id)

                # Add BetweenFactor between camera and tag
                self.graph.add(
                    gtsam.BetweenFactorPose3(
                        cam_key,
                        tag_key,
                        tag_pose_cam,
                        self.noise_model
                    )
                )

                # Add initial guess for tag pose if not already
                if not self.initial_values.exists(tag_key):
                    # Transform tag pose from camera frame to world frame
                    cam_pose = self.initial_values.atPose3(cam_key)
                    world_T_tag = cam_pose.compose(tag_pose_cam)
                    self.initial_values.insert(tag_key, world_T_tag)

        print("Factor graph built.")
    
    def solve_graph(self):
        '''
        The main function, collecting the apriltag estimations, constructing the factor graph, optimizing the MLE, and plotting the output.
        '''
        self.collect_tags()
        self.construct_graph()

        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_values)
        result = optimizer.optimize()
        
        print("Optimization complete.")
        
        self.plot_poses(result)

    def plot_poses(self, result):
        '''
        Plots the optimized results of the MLE for the apriltag poses and camera poses.
        '''
        num_cams = self.num_imgs
        tag_ids=[tag_id for tag_id, _ in self.tags[0]]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        cam_x, cam_y, cam_z = [], [], []
        tag_x, tag_y, tag_z = [], [], []

        for i in range(num_cams):
            key = gtsam.symbol('x', i)
            if result.exists(key):
                pose = result.atPose3(key)
                t = pose.translation()
                cam_x.append(t[0]); cam_y.append(t[1]); cam_z.append(t[2])

        for j in tag_ids:
            key = gtsam.symbol('t', j)
            if result.exists(key):
                pose = result.atPose3(key)
                t = pose.translation()
                tag_x.append(t[0]); tag_y.append(t[1]); tag_z.append(t[2])

        # Plot cameras (blue)
        ax.scatter(cam_x, cam_y, cam_z, c='b', marker='o', label='Cameras')

        # Plot tags (red)
        ax.scatter(tag_x, tag_y, tag_z, c='r', marker='^', label='Tags')

        # Draw coordinate axes
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        ax.set_title('Jointly Estimated Camera and Tag Poses')
        plt.show()
    
vis_slam = VisualSLAM(num_images)
vis_slam.solve_graph()
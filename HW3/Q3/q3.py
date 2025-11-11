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
        Collects all the apriltags from all images.
        '''
        for i in range(self.num_imgs):
            self.path = f'HW3/Q2/vslam/frame_{i}.jpg'
            self.tags.append(self.detect_tags())
        print("Collection Complete.\n")
        # print(self.tags)
    
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
            tag_center = tag['center']
            tag_corners = tag['lb-rb-rt-lt']
            
            # Define the 3D coordinates of the tag corners in the tag's own coordinate frame
            # For a square tag of size 1x1, assuming the tag is flat (z = 0)
            tag_size = 0.15  # Example tag size in meters (adjust as needed)
            tag_3d_points = np.array([
                [0, 0, 0],  # Bottom-left corner
                [tag_size, 0, 0],  # Bottom-right corner
                [tag_size, tag_size, 0],  # Top-right corner
                [0, tag_size, 0],  # Top-left corner
            ], dtype=np.float32)

            # Camera intrinsic parameters
            K = np.array([
                [1482.26, 0, 549.54],
                [0, 1476.49, 936.78],
                [0, 0, 1]
            ], dtype=np.float32)

            tag_2d_points = np.array(tag['lb-rb-rt-lt'], dtype=np.float32)
            
            # Use this matrix in functions like solvePnP:
            _, rvec, tvec = cv2.solvePnP(tag_3d_points, tag_2d_points, K, None)
            R, _ = cv2.Rodrigues(rvec)
            # print(R, tvec)
            tags.append([id, R, tvec])

            if not self.initial_values.exists(id):  # Check if the key already exists
                tag_pose = gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(tvec[0][0], tvec[1][0], tvec[2][0]))
                self.initial_values.insert(id, tag_pose)  # Insert the tag pose only if not present

        return tags

    def construct_graph(self):
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
        self.graph.add(gtsam.PriorFactorPose3(1, gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 0, 0)), prior_noise))
        
        for i in range(self.num_imgs):
            for tag_id, R, tvec in self.tags[i]:
                camera_pose = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 0, 0))  # Replace with actual camera pose if available
                self.initial_values.insert(i + 1, camera_pose)

                # Create relative pose (BetweenFactor) between camera pose X_i and tag pose Y_j
                tag_pose = gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(tvec[0][0], tvec[1][0], tvec[2][0]))
                relative_pose = gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(tvec[0][0], tvec[1][0], tvec[2][0]))
                self.graph.add(gtsam.BetweenFactorPose3(i + 1, tag_id, relative_pose, self.noise_model))
        
    def solve_graph(self):

        self.collect_tags()
        self.construct_graph()

        # Create the optimizer (using Levenberg-Marquardt optimizer)
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_values)

        # Optimize the factor graph
        result = optimizer.optimize()

        camera_poses = []
        tag_poses = []

        # Extract the optimized camera poses
        for i in range(1, self.num_imgs + 1):
            camera_poses.append(result.atPose3(i).translation())  # Extract the camera pose

        # Extract the optimized tag poses
        for tag_id in range(1, len(self.tags) + 1):
            tag_poses.append(result.atPose3(tag_id).translation())  # Extract the tag pose

        self.plot_poses(camera_poses, tag_poses)

    def plot_poses(self, camera_poses, tag_poses):
        '''
        Plot the optimized camera and tag poses.
        '''
        # Convert camera and tag poses to numpy arrays
        camera_poses = np.array(camera_poses)
        tag_poses = np.array(tag_poses)

        # Plotting in 3D using matplotlib
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot camera poses (blue)
        ax.scatter(camera_poses[:, 0], camera_poses[:, 1], camera_poses[:, 2], color='b', label='Camera Poses')

        # Plot tag poses (red)
        ax.scatter(tag_poses[:, 0], tag_poses[:, 1], tag_poses[:, 2], color='r', label='Tag Poses')

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera and Tag Poses')

        # Show the legend
        ax.legend()

        # Show the plot
        plt.show()
    
vis_slam = VisualSLAM(num_images)
vis_slam.solve_graph()
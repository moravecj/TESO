import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

class CarlaDatasetDrift:
    BASELINE = 1.0
    WIDTH, HEIGHT = 1024, 512
    DATASET_SIZE = 1000
    FOV = 70

    idx = 0
    epoch = 0
    
    def __init__(self, base_path):
        self.base_path = base_path

        self.K = np.array([[(self.WIDTH * 0.5) / np.tan(0.5 * self.FOV * np.pi / 180.0), 0, self.WIDTH / 2],\
            [0, (self.WIDTH * 0.5) / np.tan(0.5 * self.FOV * np.pi / 180.0), self.HEIGHT / 2],\
            [0, 0, 1]])

        R1 = np.eye(3)
        R2 = np.eye(3)
        t1 = -R1.T @ np.array([-self.BASELINE/2, 0, 0]).reshape((3,))
        t2 = -R2.T @ np.array([self.BASELINE/2, 0, 0]).reshape((3,))
        R_lid = np.eye(3)
        t_lid = -R_lid.T @ np.array([0, 0, 0]).reshape((3,))
        
        self.T21 = np.eye(4)
        self.T21[:3, :3] = R2 @ R1.T
        self.T21[:3, -1] = -R2 @ R1.T @ t1 + t2

        self.T1_lid = np.eye(4)
        self.T1_lid[:3, :3] = R1 @ R_lid.T
        self.T1_lid[:3, -1] = -R1 @ R_lid.T @ t_lid + t1

        self.T2_lid = np.eye(4)
        self.T2_lid[:3, :3] = R2 @ R_lid.T
        self.T2_lid[:3, -1] = -R2 @ R_lid.T @ t_lid + t2
        
    def readData(self, pcl_load=False, depth_map_load=False):
        img0 = cv2.cvtColor(cv2.imread(os.path.join(self.base_path, str(self.idx).zfill(3) + '_left.jpg')), cv2.COLOR_BGR2GRAY)
        img1 = cv2.cvtColor(cv2.imread(os.path.join(self.base_path, str(self.idx).zfill(3) + '_right.jpg')), cv2.COLOR_BGR2GRAY)
        img1_drift = cv2.cvtColor(cv2.imread(os.path.join(self.base_path, str(self.idx).zfill(3) + '_right_drift.jpg')), cv2.COLOR_BGR2GRAY)
        pcl, img0_depth = None, None
        if pcl_load:
            pcl = np.load(os.path.join(self.base_path, str(self.idx).zfill(3) + '.npy'))
        if depth_map_load:
            img0_depth = cv2.imread(os.path.join(self.base_path, str(self.idx).zfill(3) + '_depth.png')).astype(np.float32)
            img0_depth = ((img0_depth[:, :, 0] * 256 * 256 + img0_depth[:, :, 1] * 256 +  img0_depth[:, :, 2]) / (256*256*256 - 1)) * 1000
            img0_depth[img0_depth > 100] = np.nan

        self.idx += 1

        if self.idx >= self.DATASET_SIZE:
            self.epoch += 1
            self.idx = 1

        return img0, img1, img1_drift, pcl, img0_depth
    
    def read(self, idx, pcl_load=False, depth_map_load=False):
        img0 = cv2.cvtColor(cv2.imread(os.path.join(self.base_path, str(idx).zfill(3) + '_left.jpg')), cv2.COLOR_BGR2GRAY)
        img1 = cv2.cvtColor(cv2.imread(os.path.join(self.base_path, str(idx).zfill(3) + '_right.jpg')), cv2.COLOR_BGR2GRAY)
        img1_drift = cv2.cvtColor(cv2.imread(os.path.join(self.base_path, str(idx).zfill(3) + '_right_drift.jpg')), cv2.COLOR_BGR2GRAY)
        pcl, img0_depth = None, None
        if pcl_load:
            pcl = np.load(os.path.join(self.base_path, str(idx).zfill(3) + '.npy'))
        if depth_map_load:
            img0_depth = cv2.imread(os.path.join(self.base_path, str(idx).zfill(3) + '_depth.png')).astype(np.float32)
            img0_depth = ((img0_depth[:, :, 0] * 256 * 256 + img0_depth[:, :, 1] * 256 +  img0_depth[:, :, 2]) / (256*256*256 - 1)) * 1000
            img0_depth[img0_depth > 100] = np.nan

        return img0, img1, img1_drift, pcl, img0_depth

    def getFundamentalMatrix(self):
        T = self.T21
        K = self.K.copy()
        tx = np.array([[0, -T[2, 3], T[1, 3]], [T[2, 3], 0, -T[0, 3]], [-T[1, 3], T[0, 3], 0]])
        return np.linalg.inv(K[:3, :3]).T @ T[:3, :3] @ tx @ np.linalg.inv(K[:3, :3])
    
    def getEsentialMatrix(self):
        T = self.T21.copy()
        
        tx = np.array([[0, -T[2, 3], T[1, 3]], [T[2, 3], 0, -T[0, 3]], [-T[1, 3], T[0, 3], 0]])
        return  tx @ T[:3, :3]

    def projectLidarToImage(self, pcl, T_cam_to_lid):
        pcl2d = self.K @ (T_cam_to_lid[:3,:3] @ pcl.T + T_cam_to_lid[:3,3:4])
        pcl2d_idx = pcl2d[2, :] > 0
        pcl2d /= pcl2d[2, :]
        pcl2d_idx = np.logical_and(pcl2d_idx, np.logical_and(np.logical_and(pcl2d[1, :] >= 0, pcl2d[1, :] < self.HEIGHT), np.logical_and(pcl2d[0, :] >= 0, pcl2d[0, :] < self.WIDTH)))

        return pcl2d[:, pcl2d_idx], pcl2d_idx
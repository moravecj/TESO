import numpy as np
from matplotlib import pyplot as plt

from include.CarlaDatasetDrift import CarlaDatasetDrift

data_path = '/path/to/carla_drift'

dat = CarlaDatasetDrift('{}/data_000'.format(data_path))
img0, img1, img1_drift, pcl, img0_depth = dat.read(0, pcl_load = True, depth_map_load = True)
pcl2d, pcl2d_idx = dat.projectLidarToImage(pcl, dat.T1_lid)
plt.imshow(img0_depth)
plt.scatter(pcl2d[0, :], pcl2d[1, :], 5, c=np.linalg.norm(pcl[pcl2d_idx, :], axis=1), cmap='jet')
plt.show()
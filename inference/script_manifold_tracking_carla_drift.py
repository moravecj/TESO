import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2
import faiss

from include.FeatureExtraction import FeatureExtractor
from include.CarlaDatasetDrift import CarlaDatasetDrift
from include.EssentialMatrixManifold import EssentialMatrixManifold
from include.SGDSchaulEssentialManifold import SGDSchaulEssentialManifold

data_path = '/path/to/carla_drift'

sigm = 0.001
K_cnt = 5

fe = FeatureExtractor()

worker_id = 0
worker_cnt = 1

for seq_id in range(worker_id, 156, worker_cnt):
    print('Sequence: ', seq_id)
    errs = np.zeros((10000, 6))
    drift = np.loadtxt('{}/drifts/{}.txt'.format(data_path, str(seq_id).zfill(3)))
    file_name = 'data_{}'.format(str(seq_id).zfill(3))
    dat = CarlaDatasetDrift('{}/{}'.format(data_path, file_name))

    cnt = 0
    E = dat.getEsentialMatrix()
    U_0, S_0, Vh_0 = np.linalg.svd(E)
    U_0 = np.linalg.det(U_0) * U_0
    Vh_0 = np.linalg.det(Vh_0) * Vh_0
    emm = EssentialMatrixManifold(sigm, U_0.copy(), Vh_0.copy())
    sgd_emm = SGDSchaulEssentialManifold(emm,sigm/4)

    while dat.epoch < 1:
        img0, img1, img1_drift, pcl, img0_depth = dat.readData(depth_map_load = True)

        pts0, des0 = fe.extract_sift(img0)
        pts1, des1 = fe.extract_sift(img1_drift)

        des0 = des0.astype(np.float32)
        des1 = des1.astype(np.float32)
        faiss.normalize_L2(des0)
        faiss.normalize_L2(des1)
        index1 = faiss.IndexFlatIP(des0.shape[1])
        index0 = faiss.IndexFlatIP(des0.shape[1])
        index1.add(des0)
        D1, all_idx1 = index1.search(des1, K_cnt)
        index0.add(des1)
        D0, all_idx0 = index0.search(des0, K_cnt)
        
        K1, K2 = dat.K, dat.K

        invK_pts0, invK_pts1 = np.linalg.inv(K1) @ pts0, np.linalg.inv(K2) @ pts1
        invK_pts0 = invK_pts0 / invK_pts0[2,:]
        invK_pts1 = invK_pts1 / invK_pts1[2,:]

        sgd_emm.update(invK_pts0, invK_pts1, all_idx0, all_idx1)
        
        R1, R2, t = cv2.decomposeEssentialMat(emm._U @ emm.E_0 @ emm._Vh)
        if np.linalg.norm(R.from_matrix(R1).as_euler('xyz')) < np.linalg.norm(R.from_matrix(R2).as_euler('xyz')):
            errs[cnt, :3] = R.from_matrix(R1).as_euler('xyz', degrees=True)
            R_pom = R1
        else:
            errs[cnt, :3] = R.from_matrix(R2).as_euler('xyz', degrees=True)
            R_pom = R2
        if t[0, 0] > 0:
            t = -t
        errs[cnt, 3:] = t[:, 0] * dat.BASELINE

        cnt += 1

    res = np.mean(np.abs(errs[:cnt, :3] + drift[:, :3]),axis=0)
    res_tr = np.mean(np.abs(np.array([dat.BASELINE, 0, 0]) + errs[:cnt, 3:]),axis=0)

    print('MAE [deg] (rx, ry, rz): ', res)
    print('MAE [mm] (tx, ty, tz): ', 1000 * res_tr)

    plt.plot(errs[:cnt, 0], 'r')
    plt.plot(errs[:cnt, 1], 'g')
    plt.plot(errs[:cnt, 2], 'b' )
    plt.plot(-drift[:, 0], 'r--')
    plt.plot(-drift[:, 1], 'g--')
    plt.plot(-drift[:, 2], 'b--' )
    plt.grid(visible=True)
    plt.legend(['Rx TESO Tracked', 'Ry TESO Tracked', 'Rz TESO Tracked', 'Rx Drift', 'Ry Drift', 'Rz Drift'], fontsize=14, loc='lower left', ncol=2)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.ylabel('Rotation [°]', fontsize=17)
    plt.xlabel('Frame', fontsize=17)
    plt.xlim([0, 1000])
    plt.show()
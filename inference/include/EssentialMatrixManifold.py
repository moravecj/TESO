import numpy as np
from scipy.linalg import expm

class EssentialMatrixManifold:
    def __init__(self, sigm, U, Vh):
        self._sigm = sigm
        self._U = U
        self._Vh = Vh
        if np.linalg.det(self._U) < 0:
            self._U = -self._U
        if np.linalg.det(self._Vh) < 0:
            self._Vh = -self._Vh
        self.E_0 = np.eye(3)
        self.E_0[2, 2] = 0

    def omega_1_map(self, x):
        return 1/np.sqrt(2) * np.array([[0, -x[2]/np.sqrt(2), x[1]], [x[2]/np.sqrt(2), 0, -x[0]], [-x[1], x[0], 0]])
    
    def omega_2_map(self, x):
        return 1/np.sqrt(2) * np.array([[0, x[2]/np.sqrt(2), x[4]], [-x[2]/np.sqrt(2), 0, -x[3]], [-x[4], x[3], 0]])
    
    def map(self, x):
        omega_1 = self.omega_1_map(x)
        omega_2 = self.omega_2_map(x)
        return self._U @ expm(omega_1) @ self.E_0 @ expm(-omega_2) @ self._Vh
    
    def d_omega_1_map(self, i):
        x = np.zeros((5,))
        x[i] = 1
        return self.omega_1_map(x)
    
    def d_omega_2_map(self, i):
        x = np.zeros((5,))
        x[i] = 1
        return self.omega_2_map(x)
    
    def d_map(self, i):
        d_omega_1 = self.d_omega_1_map(i)
        d_omega_2 = self.d_omega_2_map(i)
        return self._U @ (d_omega_1 @ self.E_0 - self.E_0 @ d_omega_2) @ self._Vh
    
    def epi_err(self, kp1, kp2, all_idx1, all_idx2, x = np.zeros(5,)):
        err = 0

        E = self.map(x)
        Ex = (E @ kp1[:, all_idx1[:, 0] != -1])
        all_idx1 = all_idx1[all_idx1[:, 0] != -1, :]
        for i in range(all_idx1.shape[1]):
            num = (kp2[0, all_idx1[:, i]] * Ex[0, :] + kp2[1, all_idx1[:, i]] * Ex[1, :] + kp2[2, all_idx1[:, i]] * Ex[2, :])**2
            err -= np.sum(np.sum(np.exp(-num/(2 * self._sigm ** 2))))

        ETy = (E.T @ kp2[:, all_idx2[:, 0] != -1])
        all_idx2 = all_idx2[all_idx2[:, 0] != -1, :]
        for i in range(all_idx2.shape[1]):
            num = (kp1[0, all_idx2[:, i]] * ETy[0, :] + kp1[1, all_idx2[:, i]] * ETy[1, :] + kp1[2, all_idx2[:, i]] * ETy[2, :])**2
            err -= np.sum(np.sum(np.exp(-num/(2 * self._sigm ** 2))))
        return err
    
    def grad_hess_diag_epi_err(self, kp1, kp2, all_idx1, all_idx2):
        grads = np.zeros((5,))
        hess = np.zeros((5,))
        for j in range(5):
            E = self.map(np.zeros((5,)))
            DE = self.d_map(j)
            
            Ex = (E @ kp1[:, all_idx1[:, 0] != -1])
            DEx = (DE @ kp1[:, all_idx1[:, 0] != -1])
            for i in range(all_idx1.shape[1]):
                num = (kp2[0, all_idx1[:, i]] * Ex[0, :] + kp2[1, all_idx1[:, i]] * Ex[1, :] + kp2[2, all_idx1[:, i]] * Ex[2, :])**2
                num_d = (kp2[0, all_idx1[:, i]] * DEx[0, :] + kp2[1, all_idx1[:, i]] * DEx[1, :] + kp2[2, all_idx1[:, i]] * DEx[2, :]) * \
                    (kp2[0, all_idx1[:, i]] * Ex[0, :] + kp2[1, all_idx1[:, i]] * Ex[1, :] + kp2[2, all_idx1[:, i]] * Ex[2, :])
                num_d_part = (kp2[0, all_idx1[:, i]] * DEx[0, :] + kp2[1, all_idx1[:, i]] * DEx[1, :] + kp2[2, all_idx1[:, i]] * DEx[2, :])
                grads[j] -= np.sum(np.sum(-(1/self._sigm**2) * np.exp(-num/(2 * self._sigm ** 2)) * num_d))
                                
                hess[j] -= np.sum(np.sum((1/self._sigm**4) * np.exp(-num/(2 * self._sigm ** 2)) * num_d * num_d - (1/self._sigm**2) * np.exp(-num/(2 * self._sigm ** 2)) * num_d_part**2))
                
            ETy = (E.T @ kp2[:, all_idx2[:, 0] != -1])
            DETy = (DE.T @ kp2[:, all_idx2[:, 0] != -1])
            for i in range(all_idx2.shape[1]):
                num = (kp1[0, all_idx2[:, i]] * ETy[0, :] + kp1[1, all_idx2[:, i]] * ETy[1, :] + kp1[2, all_idx2[:, i]] * ETy[2, :])**2
                num_d = (kp1[0, all_idx2[:, i]] * DETy[0, :] + kp1[1, all_idx2[:, i]] * DETy[1, :] + kp1[2, all_idx2[:, i]] * DETy[2, :]) *\
                    (kp1[0, all_idx2[:, i]] * ETy[0, :] + kp1[1, all_idx2[:, i]] * ETy[1, :] + kp1[2, all_idx2[:, i]] * ETy[2, :])      
                num_d_part = (kp1[0, all_idx2[:, i]] * DETy[0, :] + kp1[1, all_idx2[:, i]] * DETy[1, :] + kp1[2, all_idx2[:, i]] * DETy[2, :])        
                grads[j] -= np.sum(np.sum(-(1/self._sigm**2) * np.exp(-num/(2 * self._sigm ** 2)) * num_d))

                hess[j] -= np.sum(np.sum((1/self._sigm**4) * np.exp(-num/(2 * self._sigm ** 2)) * num_d * num_d - (1/self._sigm**2) * np.exp(-num/(2 * self._sigm ** 2)) * num_d_part**2))

        hess = np.diag(np.abs(hess))
        return grads, hess

    def update(self, upd):
        self._U = self._U @ expm(self.omega_1_map(upd))
        self._Vh = expm(-self.omega_2_map(upd)) @ self._Vh
        
        if np.linalg.det(self._U) < 0:
            self._U = -self._U
        if np.linalg.det(self._Vh.T) < 0:
            self._Vh = -self._Vh
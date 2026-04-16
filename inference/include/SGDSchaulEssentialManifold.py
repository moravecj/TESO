import numpy as np

from include.EssentialMatrixManifold import EssentialMatrixManifold

class SGDSchaulEssentialManifold:
    def __init__(self, emm : EssentialMatrixManifold, upd_bnd = 0.001, m_max = 10, burn_in = 10):
        self.BURN_IN = burn_in
        self.UPD_BND = upd_bnd
        self.M_MAX = m_max
        self.emm_ = emm
        
        self.m_ = np.ones((5,))
        self.m_grads_ = np.zeros((5,))
        self.m_hes_ = np.zeros((5,5))
        self.m_v_ = np.zeros((5,))

    def update(self, invK_pts0, invK_pts1, all_idx0, all_idx1):
        upd = np.zeros((5,))
        grads, hess = self.emm_.grad_hess_diag_epi_err(invK_pts0, invK_pts1, all_idx0, all_idx1)

        self.m_grads_ = (self.m_ - 1) / self.m_ * self.m_grads_ + 1 / self.m_ * grads
        self.m_v_ = (self.m_ - 1) / self.m_ * self.m_v_ + 1 / self.m_ * (grads ** 2)
        self.m_hes_ = (self.m_ - 1) / self.m_ * self.m_hes_ + 1 / self.m_ * hess
        lr = ((self.m_grads_ ** 2) / (self.m_v_ + 0.0000001))

        if self.BURN_IN == 1:
            self.m_ = (1 - ((self.m_grads_ ** 2) / (self.m_v_ + 0.0000001))) * self.m_ + 1
            self.m_[self.m_ > self.M_MAX] = self.M_MAX
            self.m_[self.m_ < 1.0] = 1.0
            
            upd = -lr * np.linalg.inv(self.m_hes_) @ grads
            upd_sgn = upd / (np.abs(upd) + 0.0000001)
            upd = (np.abs(upd) <= self.UPD_BND) * upd + (np.abs(upd) > self.UPD_BND) * upd_sgn * self.UPD_BND
            self.emm_.update(upd)
        else:
            self.m_ += 1
            self.BURN_IN -= 1

        return upd

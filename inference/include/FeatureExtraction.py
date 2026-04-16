import numpy as np
import cv2

class FeatureExtractor:
    def __init__(self, nf=1000):
        self.orb = cv2.ORB_create(nfeatures=nf)
        self.sift = cv2.SIFT_create(nfeatures=nf)
        self.brisk = cv2.BRISK_create()

    def extract_orb(self, img):
        kp, des = self.orb.detectAndCompute(img, None)
        pts = np.asarray([[p.pt[0], p.pt[1], 1] for p in kp]).T.astype(np.float32)

        return pts, des
    
    def extract_brisk(self, img):
        kp, des = self.brisk.detectAndCompute(img, None)
        pts = np.asarray([[p.pt[0], p.pt[1], 1] for p in kp]).T.astype(np.float32)

        return pts, des
 
    def extract_sift(self, img):
        kp, des = self.sift.detectAndCompute(img,None)
        pts = np.asarray([[p.pt[0], p.pt[1], 1] for p in kp]).T.astype(np.float32)

        return pts, des
    
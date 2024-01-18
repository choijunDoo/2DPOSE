import torch
import numpy as np
import cv2
import pycocotools.mask as mask_util


def annToMask(segm, h, w):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = mask_util.frPyObjects(segm, h, w)
        rle = mask_util.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = mask_util.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann['segmentation']

    mask = mask_util.decode(rle)
    return mask


def getDensePoseMask(Polys):
    MaskGen = np.zeros([256,256])
    for i in range(1,15):
        if(Polys[i-1]):
            current_mask = mask_util.decode(Polys[i-1])
            MaskGen[current_mask>0] = i
    return MaskGen


def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def numpyPSNR(tar_img, prd_img):
    imdff = np.float32(prd_img) - np.float32(tar_img)
    rmse = np.sqrt(np.mean(imdff**2))
    ps = 20*np.log10(255/rmse)
    return ps

class HeatmapGenerator():
    def __init__(self, output_res, num_joints, sigma=-1):
        self.output_res = output_res
        self.num_joints = num_joints
        if sigma < 0:
            sigma = self.output_res/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, joints, joint_valids, people_valids):
        hms = np.zeros((self.num_joints, self.output_res, self.output_res),
                       dtype=np.float32)
        sigma = self.sigma

        for i, joints_per_person in enumerate(joints):
            if people_valids[i] > 0:
                for idx, pt in enumerate(joints_per_person):
                    if joint_valids[i][idx] > 0:
                        x, y = int(pt[0]), int(pt[1])
                        if x < 0 or y < 0 or x >= self.output_res or y >= self.output_res:
                            continue

                        ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                        br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                        c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                        a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                        cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                        aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                        hms[idx, aa:bb, cc:dd] = np.maximum(hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms
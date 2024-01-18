import os
import torch
import random
import math
import numpy as np
import os.path as osp
import glob
import json
from pycocotools.coco import COCO

from core.config import cfg
from utils.funcs_utils import process_bbox, world2cam, cam2pixel, split_into_chunks, transform_joint_to_other_db
from dataset.base_dataset import BaseDataset


class MSCOCO(BaseDataset):
    def __init__(self, transform, data_split):
        super(MSCOCO, self).__init__()
        self.transform = transform
        self.data_split = data_split
        self.img_dir = osp.join('data', 'MSCOCO', 'images')
        self.annot_path = osp.join('data', 'MSCOCO', 'annotations')

        self.joint_set = {
            'name': 'COCO',
            'joint_num': 17,
            'joints_name': ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle'),
            'flip_pairs': ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)),
            'skeleton': ((1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 6), (11, 12))
        }
        self.root_joint_idx = self.joint_set['joints_name'].index('Nose')
        self.datalist = self.load_data()
    
    def add_pelvis_and_neck(self, joint_coord):
        lhip_idx = self.joint_set.joints_name.index('L_Hip')
        rhip_idx = self.joint_set.joints_name.index('R_Hip')
        pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
        pelvis = pelvis.reshape((1, -1))

        lshoulder_idx = self.joint_set.joints_name.index('L_Shoulder')
        rshoulder_idx = self.joint_set.joints_name.index('R_Shoulder')
        neck = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
        neck = neck.reshape((1, -1))

        joint_coord = np.concatenate((joint_coord, pelvis, neck))
        return joint_coord

    def load_data(self):            
        db = COCO(osp.join(self.annot_path, 'coco_wholebody_' + self.data_split + '_v1.0.json'))

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']

            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.img_dir, self.data_split + '2017', img['file_name'])

            if ann['iscrowd'] or (ann['num_keypoints'] == 0):
                continue

            # bbox = process_bbox(ann['bbox'], (img['height'], img['width']), cfg.HMR.input_img_shape, expand_ratio=cfg.DATASET.bbox_expand_ratio) 
            # if bbox is None: continue
            
            joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(17,-1)
            # joint_img = transform_joint_to_other_db(joint_img, self.openpose_joints_name, self.joint_set['joints_name'])
            joint_img_valid = (joint_img[:,[-1]] > 0).astype(dtype=np.float32)
            joint_img = np.concatenate((joint_img[:,:2], joint_img_valid), axis=-1).astype(np.float32)

            # joint_cam = np.array(ann['joint_cam'], dtype=np.float32).reshape(-1, 3)
            
            # cam_param = {k: np.array(v, dtype=np.float32) for k,v in img['cam_param'].items()}
            # smpl_param = {k: np.array(v, dtype=np.float32) if isinstance(v, list) else v for k,v in ann['smpl_param'].items()}

            # seq_name = img['sequence'] + '_' + str(ann['person_id'])

            datalist.append({
                'ann_id': aid,
                'img_id': image_id,
                'img_path': img_path,
                'img_shape': (img['height'], img['width']),
                'joint_img': joint_img
            })

        # self.seq_names = np.unique(np.array(seq_names))
        # vid_indices = split_into_chunks(np.array(seq_names), cfg.MD.seqlen, self.stride)
        # return datalist, vid_indices
        return datalist
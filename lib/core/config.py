import os
import os.path as osp
import shutil

import yaml
from easydict import EasyDict as edict
import datetime

def init_dirs(dir_list):
    for dir in dir_list:
        if os.path.exists(dir) and os.path.isdir(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)

cfg = edict()

""" Directory """
# cfg.cur_dir = osp.dirname(os.path.abspath(__file__))
# cfg.root_dir = osp.join(cfg.cur_dir, '../../')

cfg.root_dir = os.getcwd()
cfg.data_dir = osp.join(cfg.root_dir, 'data')
KST = datetime.timezone(datetime.timedelta(hours=9))
save_folder = 'exp_' + str(datetime.datetime.now(tz=KST))[5:10]
save_folder = save_folder.replace(" ", "_")
save_folder_path = 'experiment/{}'.format(save_folder)

cfg.output_dir = osp.join(cfg.root_dir, save_folder_path)
cfg.vis_dir = osp.join(cfg.output_dir, 'vis')
cfg.log_dir = osp.join(cfg.output_dir, 'log')
cfg.checkpoint_dir = osp.join(cfg.output_dir, 'checkpoints')

print("Experiment Data on {}".format(cfg.output_dir))
init_dirs([cfg.output_dir, cfg.log_dir, cfg.vis_dir, cfg.checkpoint_dir])


""" Dataset """
cfg.DATASET = edict()
cfg.DATASET.train_list = []
cfg.DATASET.validation_list = []
cfg.DATASET.train_partition = [1.0]
cfg.DATASET.num_joint = 24
cfg.DATASET.maximum_people = 2
cfg.DATASET.test_list = []
cfg.DATASET.make_same_len = True
cfg.DATASET.select_seq_name = ''
cfg.DATASET.workers = 0
cfg.DATASET.random_seed = 123
cfg.DATASET.bbox_expand_ratio = 1.0

""" Model - HMR """
cfg.HMR = edict()
cfg.HMR.input_img_shape = (512, 512)
cfg.HMR.img_feat_shape = (32, 32)
cfg.HMR.weight_path = ''

# """ Model - MP """
# cfg.MD = edict()
# cfg.MD.hidden_dim = 144
# cfg.MD.seqlen = 49
# cfg.MD.mid_frame = 24
# cfg.MD.num_layers = 4
# cfg.MD.weight_path = ''

""" Train Detail """
cfg.TRAIN = edict()
cfg.TRAIN.total_cycle = 50
cfg.TRAIN.batch_size = 4
cfg.TRAIN.shuffle = True
cfg.TRAIN.scheduler = 'step'
cfg.TRAIN.lr = 5.0e-5
cfg.TRAIN.min_lr = 1e-6
cfg.TRAIN.lr_step = [20]
cfg.TRAIN.lr_factor = 0.1
cfg.TRAIN.optimizer = 'adam'
cfg.TRAIN.momentum = 0
cfg.TRAIN.weight_decay = 0
cfg.TRAIN.print_freq = 10
cfg.TRAIN.beta1 = 0.5
cfg.TRAIN.beta2 = 0.9

cfg.TRAIN.pose_loss_weight = 0.0
cfg.TRAIN.shape_loss_weight = 0.0
cfg.TRAIN.joint_loss_weight = 0.0
cfg.TRAIN.proj_loss_weight = 0.0
cfg.TRAIN.prior_loss_weight = 0.0

cfg.TRAIN.md_loss_weight = 0.0

""" Augmentation """
cfg.AUG = edict()
cfg.AUG.scale_factor = 0
cfg.AUG.rot_factor = 0
cfg.AUG.shift_factor = 0
cfg.AUG.color_factor = 0
cfg.AUG.blur_factor = 0
cfg.AUG.flip = False
cfg.AUG.pose_noise_ratio = 0

""" Validation Detail """
cfg.VALIDATION = edict()
cfg.VALIDATION.batch_size = 4
cfg.VALIDATION.shuffle = False
cfg.VALIDATION.print_freq = 10
cfg.VALIDATION.vis = False

""" Test Detail """
cfg.TEST = edict()
cfg.TEST.batch_size = 4
cfg.TEST.shuffle = False
cfg.TEST.print_freq = 10
cfg.TEST.vis = False

""" CAMERA """
cfg.CAMERA = edict()
cfg.CAMERA.focal = (5000, 5000)
cfg.CAMERA.princpt = (cfg.HMR.input_img_shape[1]/2, cfg.HMR.input_img_shape[0]/2)
cfg.CAMERA.camera_3d_size = 2.5


def _update_dict(k, v):
    for vk, vv in v.items():
        if vk in cfg[k]:
            cfg[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))

def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, yaml.SafeLoader))
        for k, v in exp_config.items():
            if k in cfg:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        cfg[k][0] = (tuple(v))
                    else:
                        cfg[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))

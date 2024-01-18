import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import copy
import os.path as osp

import warnings
from itertools import repeat
import collections.abc
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor

from models.resnet import PoseResNet
from models.module import HMRNet, MDNet, HybridEmbed, PatchEmbed, Block, trunc_normal_
from core.config import cfg
from core.logger import logger
from utils.train_utils import load_checkpoint
from utils.funcs_utils import rot6d_to_axis_angle, rot6d_to_rotmat, batch_rodrigues, rotmat_to_6d
from utils.human_models import smpl


class Decoder(nn.Module):
    def __init__(self, num_layers=4, dims=[384, 192, 96, 48], kernels=[4, 4, 4, 4]):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.dims = dims
        self.kernels = kernels

        self.deconv_layers = self._make_deconv_layer(self.num_layers)
        self.final_layer = nn.Conv2d(
            in_channels=self.dims[-1],
            out_channels=cfg.DATASET.num_joint*2,
            kernel_size=(1, 1),
            stride=1,
            padding=0
        )

    def _make_deconv_layer(self, num_layers):
        layers = []
        in_channel = 768
        for i in range(num_layers):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channel,
                    out_channels=self.dims[i],
                    kernel_size=self.kernels[i],
                    stride=2,
                    padding=1
                )
            )
            layers.append(nn.BatchNorm2d(self.dims[i], momentum=0.1))
            layers.append(nn.ReLU(inplace=True))

            in_channel = self.dims[i]
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x
    

class ViT(nn.Module):
    def __init__(self,
                 img_size=512, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=6,
                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=None, use_checkpoint=False, 
                 frozen_stages=-1, ratio=1, last_norm=True,
                 patch_padding='pad', freeze_attn=False, freeze_ffn=False,
                 ):
        super(ViT, self).__init__()
        # Protect mutable default arguments
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.patch_padding = patch_padding
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn
        self.depth = depth

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, ratio=ratio)
        num_patches = self.patch_embed.num_patches

        # since the pretraining model has class token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                )
            for i in range(depth)])

        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        # self._freeze_stages()
        self.decoder = Decoder()

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        if self.freeze_attn:
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.attn.eval()
                m.norm1.eval()
                for param in m.attn.parameters():
                    param.requires_grad = False
                for param in m.norm1.parameters():
                    param.requires_grad = False

        if self.freeze_ffn:
            self.pos_embed.requires_grad = False
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.mlp.eval()
                m.norm2.eval()
                for param in m.mlp.parameters():
                    param.requires_grad = False
                for param in m.norm2.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super().init_weights(pretrained, patch_padding=self.patch_padding)

        if pretrained is None:
            def _init_weights(m):
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

            self.apply(_init_weights)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)

        if self.pos_embed is not None:
            # fit for multiple GPU training
            # since the first element for pos embed (sin-cos manner) is zero, it will cause no difference
            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        x = self.last_norm(x)

        xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()
        out = self.decoder(xp)

        return out

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        # self._freeze_stages()


# class AdaptModel(nn.Module):
#     def __init__(self, hmr_net, md_net):
#         super(AdaptModel, self).__init__()
#         self.hmr_net = hmr_net
#         self.md_net = md_net
#         self.smpl_layer = copy.deepcopy(smpl.layer['neutral']).cuda()
#         self.init_weights()
    
#     def init_weights(self):
#         self.hmr_net.init_weights()
#         self.md_net.init_weights()
    
#     def load_weights(self, checkpoint):
#         if isinstance(checkpoint, tuple):
#             self.hmr_net.load_weights(checkpoint[0])
#             self.md_net.load_weights(checkpoint[1])
#         else:
#             self.load_state_dict(checkpoint['model_state_dict'])

#     def forward(self, inputs, mode, meta_info=None):
#         if mode == 'hmr':
#             return self.forward_hmr(inputs)
#         elif mode == 'md':
#             return self.forward_md(inputs, meta_info)
        
#     def forward_hmr(self, inputs):
#         x = inputs['img']

#         pred_pose6d, pred_shape, pred_cam = self.hmr_net(x)
#         pred_rotmat = rot6d_to_rotmat(pred_pose6d.reshape(-1,6)).reshape(-1, 24, 3, 3)
#         pred_pose = rot6d_to_axis_angle(pred_pose6d.reshape(-1,6)).reshape(-1, 72)

#         bs = x.shape[0]
#         pred_output = self.smpl_layer(betas=pred_shape, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,[0]], pose2rot=False)
#         pred_vertices = pred_output.vertices
#         pred_joint_cam = torch.matmul(torch.from_numpy(smpl.joint_regressor).cuda()[None,:,:], pred_vertices)

#         pred_vertices = pred_vertices + pred_cam[:,None,:]
#         pred_joint_cam = pred_joint_cam + pred_cam[:,None,:]

#         # project 3D coordinates to 2D space
#         x = pred_joint_cam[:,:,0] / (pred_joint_cam[:,:,2] + 1e-4) * cfg.CAMERA.focal[0] + cfg.CAMERA.princpt[0]
#         y = pred_joint_cam[:,:,1] / (pred_joint_cam[:,:,2] + 1e-4) * cfg.CAMERA.focal[1] + cfg.CAMERA.princpt[1]
#         pred_joint_proj = torch.stack((x,y),2)       
#         pred_joint_cam = pred_joint_cam - pred_joint_cam[:,smpl.root_joint_idx,None,:]

#         return {
#             'pose': pred_pose,
#             'rotmat': pred_rotmat.reshape(bs, -1),
#             'shape': pred_shape,
#             'trans': pred_cam,
#             'mesh_cam': pred_vertices,
#             'joint_cam': pred_joint_cam,
#             'joint_proj': pred_joint_proj
#         }
    
#     def forward_md(self, inputs, meta_info):
#         x = inputs['poses']
#         mask = meta_info['mask'][:,:,None]

#         x = x * mask
#         x = self.md_net(x)

#         return {
#             'poses': x,
#         }

def get_model(is_train):
    # hmr_net = HMRNet()
    # md_net = MDNet()
    # return AdaptModel(hmr_net, md_net)
    model = ViT()
    return model

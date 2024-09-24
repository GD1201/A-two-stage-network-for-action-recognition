# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from unittest.mock import patch
import torch.nn.functional as F
import torch
import torch.nn as nn

from timm.models.vision_transformer import  Block
import numpy as np
#from util.pos_embed import get_2d_sincos_pos_embed
from util.sklearn_embed import PatchEmbed,get_2d_sincos_pos_embed
from util.skeleton_Embed import SkeleEmbed
class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self,frame_nums=300,skeleton_nums=18,patch_size=20, in_chans=3,
                 embed_dim=16, depth=8, num_heads=4,
                 decoder_embed_dim=8, decoder_depth=6, decoder_num_heads=4,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        
        # -------------- patch_size=4, embed_dim=16, depth=8, num_heads=4,
        #decoder_embed_dim=8, decoder_depth=6, decoder_num_heads=4,
        #mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)------------------------------------------------------------
        # ST-MAE encoder specifics
        self.patch_embed = PatchEmbed(frame_nums,skeleton_nums,patch_size,embed_dim)
        self.patch_embed1=SkeleEmbed( in_chans,embed_dim,frame_nums,skeleton_nums,patch_size=1,t_patch_size=patch_size)
        num_patches = self.patch_embed.num_patches #19
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #1 271 512
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        #input x 64 1351 768
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
       
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
      
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim,  1*patch_size*in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        #self.pos_embed.shape 1*197*1024
        #out 197*1024 (196+1)*1024
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        #高斯分布初始化
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)  #权重和偏置
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        """
        imgs: (N, 3, H, W) 64 3 300 18 2
        x: (N, L, patch_size**2 *3)
        """
        ##把图片划分为块
        #p = self.patch_embed.patch_size[0]#1
        #assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        #h = w = imgs.shape[2] // p
        #x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))#64 3 14 16 14 16
        #x = torch.einsum('nchpwq->nhwpqc', x)
        #x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))  #64 196  16*16*3
        #input:64 3 300 18 2  ->> 64 2 300 18 1
        NM, C, T, V = x.size()
        # x = x.permute(0, 4, 3, 1, 2).contiguous()
        # x = x.view(N * M, V * C, T)
        # #x = data_bn(x)
        # x = x.view(N, M, V, C, T)
        # x = x.permute(0, 1, 3, 4, 2).contiguous()
        # x = x.view(N * M, C, T, V)
        # x=x[...,0]  ### n c t v  -> n c v t 64 3 300 25
        #batch 3 19 300
        x=x.transpose(3,2) ### 64 3 25 300

        h=x.shape[2]  ##25
        #75
        w=self.patch_embed.grid_size[1]###15
        p=int(x.shape[3]/self.patch_embed.grid_size[1])  ## 20
        x = x.reshape(shape=(x.shape[0], 3,  h,1,w ,p))### 64 3  25  1 15 20
        x = torch.einsum('nchpwq->nhwpqc', x)## 64 3 25 20 15 1 -> 64 25   15 20 1 3
        x = x.reshape(shape=(x.shape[0], h * w,1 *p* 3))  ##64  375 60
        #x=self.norm(x)
        #64*3*(19*300)

        # x=self.patch_embed.deproj(x)
        # x = x.flatten(2)
        # #64*19*300*16
        # x=x.transpose(2,1)

        #x=x[:,:,:,:,0].unsqueeze(-1)
        #64 18 300 2 1
        #x=x.transpose(1,3)
        
        #64 18 600
        #x = x.flatten(2)






        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3) 64 18 600
        imgs: (N, 3, H, W)          64  2 300 18 1
        """
        ### 还原成图片
        # x = x.reshape(shape=(x.shape[0], 18, 300, 2))
      
        # x=x.transpose(1,3)
        # # 64 2 300 18
        # x=x.transpose(2,3).unsqueeze(-1)
        #4
        p = self.patch_embed.patch_size[1]#20
        h = self.patch_embed.grid_size[0]#25
        w = self.patch_embed.grid_size[1]# 15
        #assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, 1, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        #batch 3 19 300
        ####   x = x.reshape(shape=(x.shape[0], h * w,1 *p* 3))
        x = x.reshape(shape=(x.shape[0], 3, h * 1, w * p))
        #bacth 3 300 19
        x=x.transpose(3,2)
      
        return x

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        #imge 64 196 1024
        #     18 
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


    def motion_aware_masking(self,x,data_skeleton,dynamic_ratio,static_ratio,frame_ratio,joint_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        #imge 64 196 1024
        #     18
        mask_ratio=dynamic_ratio+static_ratio
        N, L, D = x.shape  # batch, length, dim  
        # data_skeleton=data_skeleton[...,0]
        # N,C,T,V=data_skeleton.shape 
        data_skeleton=self.patch_embed1(data_skeleton)
        # data_skeleton=data_skeleton.reshape(N,T,V,C)
        N,T,V,C=data_skeleton.shape 
        # data_skeleton=self.patch_embed(data_skeleton)
        # len_keep = int(L * (1 - mask_ratio))
        len_keep_frame=int(T*(1-frame_ratio))
        len_keep_joint=int(V*(1-joint_ratio))
        # L=T*V
        # len_keep = int(L * (1 - mask_ratio))
        # x_=data_skeleton[:,0:2,:,:]
        x_=data_skeleton
        x_=x_.mean(dim=3)  ### N T V
        
        x_orig_motion = torch.zeros_like(x_)  ### 48 30 25 256
        # x_orig_motion[:, 1:, :, :] = torch.abs(data_skeleton[:, 1:, :, :] - data_skeleton[:, :-1, :, :])
        # x_orig_motion[:, 0, :, :] = x_orig_motion[:, 1, :, :]
        x_orig_motion[:,1:,:]=torch.abs(x_[:, 1:, :] -x_[:, :-1, :])
        x_orig_motion[:,0,:] = x_orig_motion[:, 1,:] 
        x_orig_motion = x_orig_motion.mean(dim=2) 
        x_orig_motion = x_orig_motion / (torch.max(x_orig_motion, dim=-1, keepdim=True).values * 0.85+ + 1e-10)
        x_orig_motion_prob_Frame = F.softmax(x_orig_motion, dim=-1)
        noise = torch.log(x_orig_motion_prob_Frame) - torch.log(-torch.log(torch.rand(N, T, device=x.device) + 1e-10) + 1e-10)  # gumble
        # x_masked_frame = torch.zeros_like(x_)
        ids_shuffle_frame = torch.argsort(
            noise, dim=1
        ) 
        # ascend: small is keep, large is remove
        ids_restore_frame= torch.argsort(ids_shuffle_frame, dim=1)
        ids_keep_frame = ids_shuffle_frame[:, :len_keep_frame]
        # ids_keep_frame=ids_shuffle_frame
        # for i in range(N):
        #     x_masked_frame[i, ids_keep_frame[i], :] = x_[i, ids_keep_frame[i], :]
        # x_masked_frame_motion_joint = torch.zeros_like(x_masked_frame)  ###   64  30  18 256
        # x_masked_frame_motion_joint=x_masked_frame.clone()
        # x_masked_frame_motion_joint1=x_masked_frame.clone()
        # x_masked_frame_motion_joint1[:, :, 1:, :] = torch.abs(x_masked_frame_motion_joint1[:, :, 1:, :] - x_masked_frame_motion_joint1[:, :, :-1, :])
        # x_masked_frame_motion_joint1[:, :, 0, :] = x_masked_frame_motion_joint1[:, :, 1, :]
        x_orig_motion_joint = torch.zeros_like(x_)
        x_orig_motion_joint[:, :, 1:] = torch.abs(x_[:, :, 1:] - x_[:, :,:-1])
        x_orig_motion_joint[:, :, 0] = x_orig_motion_joint[:, :, 1]
        x_orig_motion_joint = x_orig_motion_joint.mean(dim=1)  # 形状为 [N, VP]
        x_orig_motion_joint = x_orig_motion_joint/ (torch.max(x_orig_motion_joint, dim=-1, keepdim=True).values * 0.85 + 1e-10)
        x_masked_frame_motion_prob_joint = F.softmax(x_orig_motion_joint, dim=-1)
        noise_joint = torch.log(x_masked_frame_motion_prob_joint) - torch.log(-torch.log(torch.rand(N, V, device=x.device) + 1e-10) + 1e-10)
        ids_shuffle_joint = torch.argsort(
            noise_joint, dim=1
        ) 
        ids_restore_joint = torch.argsort(ids_shuffle_joint, dim=1)
        # keep the first subset
        # ids_keep = ids_shuffle[:, :len_keep]
        # ids_keep_joint=ids_shuffle_joint
        ids_keep_joint = ids_shuffle_joint[:, len_keep_joint:]
        mask_frame = torch.ones_like(x_)  # 创建一个全为1的massk
        # mask_frame[:,ids_keep_frame,ids_keep_joint]=0
        for sample_idx, frame_indices in enumerate(ids_keep_frame):
          mask_frame[sample_idx, frame_indices, :] = 0
        mask_frame1=torch.ones_like(x_)
        for sample_idx,joint_indices in enumerate(ids_keep_joint):
            mask_frame[sample_idx, :, joint_indices] = 1
       
        mask_frame=mask_frame.reshape(N,T*V) 
        mask = mask_frame
        x_masked=x.masked_select(~mask.bool().unsqueeze(-1)).reshape(N, len_keep_frame,-1, self.mask_token.shape[-1])
        x_masked=x_masked.view(N,-1,self.mask_token.shape[-1])
        mask=mask.reshape(N,T*V)
        
        # sort noise for each sample
        # ids_skeleton = self.skeleton_ids(data_skeleton,dynamic_ratio,static_ratio,L)
        # ids_restore = torch.argsort(ids_skeleton, dim=1)

        # # keep the first subset
        # ids_keep = ids_skeleton[:, :len_keep]
        # x_masked = torch.gather(x.cuda(), dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # # generate the binary mask: 0 is keep, 1 is remove
        # mask = torch.ones([N, L], device=x.device)
        # mask[:, :len_keep] = 0
        # # unshuffle to get the binary mask
        # mask = torch.gather(mask.cuda(), dim=1, index=ids_restore)

        return x_masked, mask, ids_restore_frame,ids_restore_joint
    # def motion_aware_masking(self,x,data_skeleton,dynamic_ratio,static_ratio):
    #     """
    #     Perform per-sample random masking by per-sample shuffling.
    #     Per-sample shuffling is done by argsort random noise.
    #     x: [N, L, D], sequence
    #     """
    #     #imge 64 196 1024
    #     #     18
    #     mask_ratio=dynamic_ratio+static_ratio
    #     N, L, D = x.shape  # batch, length, dim
    #     len_keep = int(L * (1 - mask_ratio))
       
    #     # sort noise for each sample
    #     ids_skeleton = self.skeleton_ids(data_skeleton,dynamic_ratio,static_ratio,True)
    #     ids_restore = torch.argsort(ids_skeleton, dim=1)

    #     # keep the first subset
    #     ids_keep = ids_skeleton[:, :len_keep]
    #     x_masked = torch.gather(x.cuda(), dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    #     # generate the binary mask: 0 is keep, 1 is remove
    #     mask = torch.ones([N, L], device=x.device)
    #     mask[:, :len_keep] = 0
    #     # unshuffle to get the binary mask
    #     mask = torch.gather(mask.cuda(), dim=1, index=ids_restore)

    #     return x_masked, mask, ids_restore





    def forward_encoder(self, x, mask_ratio,frame_ratio,joint_ratio):
        #data.shape N,C,T,V,M

        #image 64 3 224 224
        # embed patches out b 14*14 1024   224/16
        #sklearn b 18 576
        x_copy=x.clone()
        # N,C,T,V,M=x_copy.shape
        # T1=T//20
        #128*(270)*512
        x = self.patch_embed(x) # 24 256 18 9
        
        #64 270 512
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        N,L,D=x.shape
        # masking: length -> length * mask_ratio
        #self.uniform_masking(x,mask_ratio)
        #x b 64 512
        x, mask, ids_restore_frame,ids_restore_joint =self.motion_aware_masking(x, x_copy,mask_ratio-0.5,0.5,frame_ratio,joint_ratio)#self.uniform_masking(x,mask_ratio)#self.random_masking(x, mask_ratio)#self.motion_aware_masking(x, x_copy,mask_ratio-0.5,0.5)#self.random_masking(x, mask_ratio)#self.motion_aware_masking(x, x_copy,mask_ratio-0.5,0.5)#self.random_masking(x, mask_ratio)#self.motion_aware_masking(x, x_copy,mask_ratio-0.25,0.25)#self.uniform_masking(x,mask_ratio)#self.random_masking(x, mask_ratio)##self.motion_aware_masking(x, x_copy,mask_ratio-0.25,0.25)#self.random_masking(x, mask_ratio)#self.motion_aware_masking(x, x_copy,mask_ratio-0.25,0.25)#self.random_masking(x, mask_ratio)#self.dynamic_masking(x, x_copy,0.25,mask_ratio-0.25)##self.dynamic_masking(x, x_copy,0.35,mask_ratio-0.35) #self.uniform_masking(x,mask_ratio)
        #x 64 53 512  x, mask, ids_restore
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        #imge 64 1 1024
        #sk   64 1  768
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        #64 1351 768
        x = torch.cat((cls_tokens.cuda(), x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
        #
            x = blk(x)
        x = self.norm(x)
        # x=x.view(N,T1,V,D)
        return x, mask, ids_restore_frame,ids_restore_joint

    def forward_decoder(self, x,mask, ids_restore_frame,ids_restore_joint):
        # embed tokens
        # if self.use_mask_tokens==True : 
        #     N = y.shape[0]
        #     # TP = self.patch_Embed.t_grid_size
        #     # VP = self.patch_Embed.grid_size
        #     TP=self.patch_embed.grid_size[1]
        #     VP=self.patch_embed.grid_size[0]
        #     # embed tokens
        #     x = self.decoder_embed(y)
        #     C = y.shape[-1]
        #     # append intra mask tokens to sequence
        #     mask_tokens = self.mask_token.repeat(N, TP * VP - x.shape[1], 1)
        #     x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        #     x = x_.view([N, TP, VP, C]) 
        #     x_restored = torch.zeros_like(x)
        #     for i in range(N):
        #         x_restored[i, ids_restore_frame[i], :,:] = x[i]  # 还原 ids_restore_frame
        #         x_restored[i, :, ids_restore_joint[i],:] = x[i] 
        #     # print(x.shape#)     
        #     # x=x.reshape(N,TP*VP,-1)   
        #     ####选择遮掩的masktoken作为x
        #     # x =  x_restored+ self.decoder_pos_embed[:, :, :VP, :] + self.decoder_temp_embed[:, :TP, :, :]  # NM, TP, VP, C
        #     # apply Transformer blocks
        #     x = x.reshape(N, TP * VP, C)
        #     # x = self.decoder_pos_embed_mask[:, 1:].masked_select(mask.bool().unsqueeze(-1)).reshape(N, -1, self.mask_token.shape[-1])
        #     # x = x + self.mask_token
        #     x=x.masked_select(mask.bool().unsqueeze(-1)).reshape(N, -1, self.mask_token.shape[-1])
        #     for i, blk in enumerate(self.decoder_blocks1):
        #         ###交叉注意力
        #         x = blk(x, y)
        N=x.shape[0]
        C=x.shape[-1]
        x = self.decoder_embed(x)
        T=self.patch_embed.grid_size[1]
        V=self.patch_embed.grid_size[0]
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], mask.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x = x_.view([N, T, V, C]) 
        x_restored = torch.zeros_like(x)
        for i in range(N):
            x_restored[i, ids_restore_frame[i], :,:] = x[i]  # 还原 ids_restore_frame
            x_restored[i, :, ids_restore_joint[i],:] = x[i]
        # x_ = torch.gather(x_, dim=1, index=mask.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x_restored=x_restored.reshape(x.shape[0],T*V,-1)
        x1=x_restored
        x = torch.cat([x1[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed
        #2261*8
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W] 64 3 300 18 2
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        #img 14*14 16*16*3
        #skeleton 19*140  3
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio,frame_ratio,joint_ratio):
        #img:->skeleton 64 3 300 18 
        
        # print(imgs.shape)
        N, C, T, V, M = imgs.shape
        imgs = imgs.permute(0, 4, 2, 3, 1).contiguous().view(N*M,C,T,V)
        latent, mask, ids_restore_frame,ids_restore_joint= self.forward_encoder(imgs, mask_ratio,frame_ratio,joint_ratio)
        #64 18 600 latent, mask, ids_restore 
        pred = self.forward_decoder(latent, mask,ids_restore_frame,ids_restore_joint)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
# #512 256
def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        frame_nums=120,skeleton_nums=25,patch_size=6, embed_dim=256, depth=8, num_heads=8,
        decoder_embed_dim=256, decoder_depth=3, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
# def mae_vit_base_patch16_dec512d8b(**kwargs):
#     model = MaskedAutoencoderViT(
#           frame_nums=180,skeleton_nums=18,patch_size=20, embed_dim=256, depth=8, num_heads=8,
#         decoder_embed_dim=256, decoder_depth=6, decoder_num_heads=8,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model
def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        skeleton_nums=19,patch_size=20, embed_dim=512, depth=8, num_heads=8,
        decoder_embed_dim=256, decoder_depth=6, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# def mae_vit_huge_patch14_dec512d8b(**kwargs):
#     model = MaskedAutoencoderViT(
#         patch_size=14, embed_dim=1280, depth=32, num_heads=16,
#         decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks

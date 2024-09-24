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

import torch
import torch.nn as nn

import timm.models.vision_transformer
from util.sklearn_embed import PatchEmbed
from util.skeleton_Embed import SkeleEmbed
from util.drop import DropPath
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, seqlen=1):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        x = self.forward_attention(q, k, v)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_attention(self, q, k, v):
        B, _, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1,2).reshape(B, N, C*self.num_heads)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., mlp_out_ratio=1.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # assert 'stage' in st_mode
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads,
                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_out_dim = int(dim * mlp_out_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       out_features=mlp_out_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, seqlen=1):
        x = x + self.drop_path(self.attn(self.norm1(x), seqlen))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
class ActionHeadFinetune(nn.Module):
    def __init__(self, dropout_ratio=0., dim_feat=256, nb_classes=60, num_joints=25, hidden_dim=2048):
        super(ActionHeadFinetune, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bn = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim_feat*num_joints, hidden_dim)
        self.fc2 = nn.Linear( hidden_dim, nb_classes)

        
    def forward(self, feat):
        '''
            Input: (N, M, T, J, C)
        '''
        N, M, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 1, 3, 4, 2)      # (N, M, T, J, C) -> (N, M, J, C, T)
        feat = feat.mean(dim=-1)
        feat = feat.reshape(N, M, -1)           # (N, M, J*C)
        feat = feat.mean(dim=1)
        feat = self.fc1(feat)
        feat = self.bn(feat)
        feat = self.relu(feat)    
        feat=self.fc2(feat)
        return feat
class VisionTransformer(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, frame_nums=120,skeleton_nums=25,patch_size=1,
                 in_chans=3,embed_dim=256, t_patch_size=4, global_pool=False,qkv_bias=True, qk_scale=None, drop_rate=0., depth=8, num_heads=8, mlp_ratio=4,
              attn_drop_rate=0., norm_layer=nn.LayerNorm,drop_path_rate=0.3,nb_classes=60,**kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        #skeleton_nums=19,patch_size=4, embed_dim=16, depth=6, num_heads=2,
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        self.global_pool = global_pool
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.max_frame=frame_nums
        # self.num_classes=nb_classes
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        # self.patch_embed = PatchEmbed(frame_nums,skeleton_nums,patch_size,kwargs['embed_dim'])
        # self.joints_embed = SkeleEmbed(in_chans, embed_dim, frame_nums, skeleton_nums, patch_size, t_patch_size)
        self.joints_embed = SkeleEmbed(in_chans,embed_dim,frame_nums,skeleton_nums,patch_size,t_patch_size)
        self.temp_embed = nn.Parameter(torch.zeros(1, frame_nums//t_patch_size, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, skeleton_nums//patch_size, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim), requires_grad=False)
        ### motion BERT 
        self.head = ActionHeadFinetune(dropout_ratio=0.3, dim_feat=embed_dim,nb_classes=nb_classes)
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        N, C, T, V, M = x.shape
        x = x.permute(0, 4, 2, 3, 1).contiguous().view(N * M, T, V, C)
        # x=x[...,0]
        # x = x.permute(0, 2, 3, 1).contiguous().view(N*M ,T, V, C)
        x = self.joints_embed(x)
        NM, TP, VP, _ = x.shape

        x = x + self.pos_embed[:, :, :VP, :] + self.temp_embed[:, :TP, :, :]
        # x = x.reshape(NM, TP * VP, -1)
        #64 270 512
        # add pos embed w/o cls token
        # x = x + self.pos_embed[:, 1:, :]
        # cls_tokens = self.cls_token.expand(NM, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # x = torch.cat((cls_tokens, x), dim=1)
        # x = x + self.pos_embed
        
        # x = self.pos_drop(x)
        # x=x[:,1:,:]
        x = x.reshape(NM, TP * VP, -1)
        for blk in self.blocks:
            x = blk(x)
        # x = x[:, 1:, :]  ##移除 cls_token
        # if self.global_pool:
        #     x = x.mean(dim=1)  # global pool without cls token
        #     outcome = self.fc_norm(x)
        #     # outcome=outcome[1]
        # else:
        #     #########(64,540,256)
        #     # x=x.reshape(N,TP*VP,)
        #     x = self.norm(x)
        # #     outcome = x[:, 0]
        x=self.norm(x)
        x = x.reshape(N, M, TP, VP, -1)
        # print(x.shape)
        x = self.head(x)
        return x

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        frame_nums=120,skeleton_nums=25,patch_size=1, embed_dim=256, depth=8, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        skeleton_nums=19,patch_size=20, embed_dim=512, depth=8, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
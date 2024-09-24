import torch
import torch.nn as nn
import numpy as np
# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0],grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, frame_nums,skeleton_nums,patch_size,embed_dim):
        super().__init__()
        #img_size = to_2tuple(img_size)
        self.patch_size=(1,patch_size)
        #stride_size=(1,1)
        #19*(300/4)=19*75 18*30
        self.num_patches = skeleton_nums*(int(frame_nums/patch_size))
        self.num_skearn=skeleton_nums
        self.grid_size=(skeleton_nums,int(frame_nums/patch_size))
        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        #self.deproj = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(1,22), stride=(1,2))
        if skeleton_nums==19 :
         self.proj = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=(1,patch_size), stride=(1,patch_size))
        else:
         self.proj = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=(1,patch_size), stride=(1,patch_size))

    def forward(self, x):
        #input X.shape:(B,C, T, V, M) eg. 64 3 300 18 2
        #output B*18*2*300
        B, C, T, V, M = x.size()
        # FIXME look at relaxing size constraints
        # assert V == self.num_skearn  , \
        #     f"Input Sklearn patch size ({V}) doesn't match model ({self.num_skearn })."
        # 64 2 300  19
        # x=x[:,0:2,:,:,0]  ##(64,120,3,,18)
        x=x[...,0]
        #2d
        #batch 3 19 300
        x=x.transpose(3,2)   ## (64,3,120,18)
        #64 3 18 30
        x=self.proj(x) #(64,)
        #64 512 18 15
        x = x.flatten(2)
        #64 512 270
        #64*19*300*16
        x=x.transpose(2,1)
        #3d vatch 3 300 19 1
        # x=x.unsqueeze(-1)
        # #   b 16 15 19 1
        # x=self.proj(x)
        # # b 16 15 19
        # x=x.transpose(3,2).squeeze(-1)
        # x = x.flatten(2)
        # x=x.transpose(2,1)
        # mean = x.mean(dim=-1, keepdim=True)
        # var = x.var(dim=-1, keepdim=True)
        # x = (x - mean) / (var + 1.e-6)**.5
        return x
# class PatchEmbed(nn.Module):
#     """ Image to Patch Embedding
#     """
#
#     def __init__(self, num_skearn=18):
#         super().__init__()
#         # img_size = to_2tuple(img_size)
#         self.patch_size = (1, 1)
#         # stride_size=(1,1)
#         self.num_patches = num_skearn
#         self.num_skearn = num_skearn
#         self.grid_size = (num_skearn, 1)
#
#         # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
#         if num_skearn == 19:
#             self.proj = nn.Conv3d(in_channels=2, out_channels=2, kernel_size=(1, 1, 1), stride=(1, 1, 1),
#                                   padding=(2, 0, 0))
#         else:
#             self.proj = nn.Conv3d(in_channels=2, out_channels=2, kernel_size=(13, 1, 1), stride=(1, 1, 1))
#
#     def forward(self, x):
#         # input X.shape:(B,C, T, V, M) eg. 64 3 300 18 2
#         # output B*18*2*300
#
#         # B, C, T, V,M = x.shape
#         N, C, T, V, M = x.size()
#
#         # FIXME look at relaxing size constraints
#         assert V == self.num_skearn, \
#             f"Input Sklearn patch size ({V}) doesn't match model ({self.num_skearn})."
#         # 64 2 18  300
#         #
#
#         # x = x.permute(0, 4, 3, 1, 2).contiguous()
#         # x = x.view(N * M, V * C, T)
#         # #x = data_bn(x)
#         # x = x.view(N, M, V, C, T)
#         # x = x.permute(0, 1, 3, 4, 2).contiguous()
#         # x = x.view(N * M, C, T, V)[:,:2,...].unsqueeze(-1)
#         # 64 2 300 19 2 1
#         x = x[:, :2, :, :, 0].unsqueeze(-1)
#         # 64 2 300 19 1 -> 64 2 304 19 1
#         x = self.proj(x).transpose(1, 3)
#         # 64 18 3*150
#         x = x.flatten(2)
#         return x
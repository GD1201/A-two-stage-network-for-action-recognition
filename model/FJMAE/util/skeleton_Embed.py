import torch
import  torch.nn as nn

class SkeleEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
            self,
            dim_in=3,
            dim_feat=256,
            num_frames=120,
            num_joints=25,
            patch_size=1,
            t_patch_size=4,
    ):
        super().__init__()
        # print(num_frames)
        # print(t_patch_size)
        print(patch_size)
        assert num_frames % t_patch_size == 0
        num_patches = (
                (num_joints // patch_size) * (num_frames // t_patch_size)
        )
        # self.input_size = (
        #     num_frames // t_patch_size,
        #     num_joints // patch_size
        # )
        print(
            f"num_joints {num_joints} patch_size {patch_size} num_frames {num_frames} t_patch_size {t_patch_size}"
        )

        self.num_joints = num_joints
        self.patch_size = patch_size

        self.num_frames = num_frames
        self.t_patch_size = t_patch_size

        self.num_patches = num_patches

        self.grid_size = num_joints // patch_size
        self.t_grid_size = num_frames // t_patch_size
        self.grid_size_embed=[self.grid_size,self.t_grid_size]
        kernel_size = [t_patch_size, patch_size]
        self.proj = nn.Conv2d(dim_in, dim_feat, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x):
        # B, C, T, V, M = x.size()
        # x=x[...,0]
        # print(x.shape)  (32,3,120,18)
        # x=x.permute(0,)
        _, T, V, _ = x.shape
        x = torch.einsum("ntsc->ncts", x)  # [N, C, T, V]

        assert (
                V == self.num_joints
        ), f"Input skeleton size ({V}) doesn't match model ({self.num_joints})."
        assert (
                T == self.num_frames
        ), f"Input skeleton length ({T}) doesn't match model ({self.num_frames})."
        x = self.proj(x)
        x = torch.einsum("ncts->ntsc", x)  # [N, T, V, C]
        return x  ### 64  30  18   256
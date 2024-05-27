import math
import einops
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_

from timm.models.layers import to_2tuple, trunc_normal_,  DropPath
from natten.functional import NATTEN2DQKRPBFunction, NATTEN2DAVFunction


class NeighborhoodAttention2D(nn.Module):
    def __init__(self, dim, kernel_size, num_heads, attn_drop=0., proj_drop=0., dilation=None):
        super().__init__()
        self.fp16_enabled = False
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.kernel_size = kernel_size
        if type(dilation) is str:
            self.dilation = None
            self.window_size = None
        else:
            self.dilation = dilation or 1
            self.window_size = self.kernel_size * self.dilation

        self.qkv = nn.Linear(dim, dim * 3)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, Hp, Wp, C = x.shape
        H, W = int(Hp), int(Wp)
        pad_l = pad_t = pad_r = pad_b = 0
        dilation = self.dilation
        window_size = self.window_size
        if window_size is None:
            dilation = max(min(H, W) // self.kernel_size, 1)
            window_size = dilation * self.kernel_size
        if H < window_size or W < window_size:
            pad_l = pad_t = 0
            pad_r = max(0, window_size - W)
            pad_b = max(0, window_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        # breakpoint()
        attn = NATTEN2DQKRPBFunction.apply(q, k, self.rpb, self.kernel_size, dilation)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = NATTEN2DAVFunction.apply(attn, v, self.kernel_size, dilation)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x)).permute(0, 3, 1, 2), None, None



class DAttentionBaseline(nn.Module):
    def __init__(self, q_size, kv_size, n_heads, n_head_channels, n_groups,
        attn_drop, proj_drop, stride,
        no_off, ksize):

        super().__init__()
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.no_off = no_off
        self.ksize = ksize
        self.stride = stride
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

        self.proj_q = nn.Conv2d(self.nc, self.nc,kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        self.rpe_table = nn.Parameter(
            torch.zeros(self.n_heads, self.q_h * 2 - 1, self.q_w * 2 - 1)
        )
        trunc_normal_(self.rpe_table, std=0.01)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2

        return ref
    
    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2

        return ref

    def forward(self, x):

        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off).contiguous()  # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        pos = (offset + reference).clamp(-1., +1.)

        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W), 
            grid=pos[..., (1, 0)], # y, x -> x, y
            mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg
                

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k) # B * h, HW, Ns
        attn = attn.mul(self.scale)


        rpe_table = self.rpe_table
        rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
        q_grid = self._get_q_grid(H, W, B, dtype, device)
        displacement = (q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)).mul(0.5)
        attn_bias = F.grid_sample(
            input=einops.rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.n_group_heads, g=self.n_groups),
            grid=displacement[..., (1, 0)],
            mode='bilinear', align_corners=True) # B * g, h_g, HW, Ns

        attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
        attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        out = out.reshape(B, C, H, W)

        y = self.proj_drop(self.proj_out(out))

        return y, pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)
    

class LayerNormProxy(nn.Module):
    def __init__(self, dim):    
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class TransformerMLPWithConv(nn.Module):
    def __init__(self, channels, expansion, drop):
        super().__init__()
        self.dim1 = channels
        self.dim2 = channels * expansion
        self.linear1 = nn.Sequential(nn.Conv2d(self.dim1, self.dim2, 1, 1, 0))
        self.drop1 = nn.Dropout(drop, inplace=True)
        self.act = nn.GELU()
        self.linear2 = nn.Sequential(nn.Conv2d(self.dim2, self.dim1, 1, 1, 0),)
        self.drop2 = nn.Dropout(drop, inplace=True)
        self.dwc = nn.Conv2d(self.dim2, self.dim2, 3, 1, 1, groups=self.dim2)
    
    def forward(self, x):        
        x = self.linear1(x)
        x = self.drop1(x)
        x = x + self.dwc(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x
    

class TransformerStage(nn.Module):
    def __init__(self, fmap_size, dim_in, dim_embed, depths, stage_spec, n_groups, heads, stride,
                 attn_drop, proj_drop, expansion, drop, drop_path_rate, 
                 ksize, nat_ksize):

        super().__init__()
        fmap_size = to_2tuple(fmap_size)
        self.depths = depths
        hc = dim_embed // heads
        self.proj = nn.Conv2d(dim_in, dim_embed, 1, 1, 0) if dim_in != dim_embed else nn.Identity()
        self.stage_spec = stage_spec

        self.layer_norms = nn.ModuleList([nn.Identity() for d in range(2 * depths)])

        self.mlps = nn.ModuleList([ TransformerMLPWithConv(dim_embed, expansion, drop) for _ in range(depths) ])
        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()
        self.layer_scales = nn.ModuleList([ nn.Identity()  for _ in range(2 * depths)])
        self.local_perception_units = nn.ModuleList([ nn.Conv2d(dim_embed, dim_embed, kernel_size=3, stride=1, padding=1, groups=dim_embed)
                for _ in range(depths)
            ])

        for i in range(depths):
            if stage_spec[i] == 'D':
                self.attns.append(
                    DAttentionBaseline(fmap_size, fmap_size, heads, 
                    hc, n_groups, attn_drop, proj_drop, 
                    stride, ksize)
                )
            elif stage_spec[i] == 'N':
                self.attns.append(
                    NeighborhoodAttention2D(dim_embed, nat_ksize, heads, attn_drop, proj_drop)
                )
            else:
                raise NotImplementedError(f'Spec: {stage_spec[i]} is not supported.')

            self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())

    def forward(self, x):
        x = self.proj(x)

        for d in range(self.depths):            
            x0 = x
            x = self.local_perception_units[d](x.contiguous())
            x = x + x0

            x0 = x
            x, _, _ = self.attns[d](self.layer_norms[2 * d](x))
            x = self.layer_scales[2 * d](x)
            x = self.drop_path[d](x) + x0
            x0 = x
            x = self.mlps[d](self.layer_norms[2 * d + 1](x))
            x = self.layer_scales[2 * d + 1](x)
            x = self.drop_path[d](x) + x0

        return x


class DAT(nn.Module):

    def __init__(self, img_size=224, patch_size=4, num_classes=1000, expansion=4,
                 dim_stem=64, dims=[64, 128, 256, 512], depths=[2, 4, 18, 2], 
                 heads=[2, 4, 8, 16], drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.2, 
                 strides=[8, 4, 2, 1],
                 stage_spec=[['N', 'D'], ['N', 'D', 'N', 'D'], 
                             ['N', 'D', 'N', 'D', 'N', 'D', 'N', 'D', 'N', 'D', 'N', 'D', 'N', 'D', 'N', 'D', 'N', 'D'], ['L', 'D']], 
                 groups=[1, 2, 4, 8],
                #  use_pes=[True, True, True, True], 
                #  dwc_pes=[False, False, False, False],
                #  fixed_pes=[False, False, False, False],
                #  no_offs=[False, False, False, False],
                #  use_dwc_mlps=[True, True, True, True],
                 ksizes=[9, 7, 5, 3],
                #  ksize_qnas=[3, 3, 3, 3],
                #  nqs=[2, 2, 2, 2],
                #  qna_activation='exp',
                 nat_ksizes=[7, 7, 7, 7],
                #  layer_scale_values=[-1,-1,-1,-1],
                #  use_lpus=[True, True, True, True]
                 ):
        
        super().__init__()
        self.patch_proj = nn.Sequential(
            nn.Conv2d(3, dim_stem // 2, 3, patch_size // 2, 1),
            LayerNormProxy(dim_stem // 2),
            nn.GELU(),
            nn.Conv2d(dim_stem // 2, dim_stem, 3, patch_size // 2, 1),
            LayerNormProxy(dim_stem)
        )

        img_size = img_size // patch_size
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        for i in range(4):
            dim1 = dim_stem if i == 0 else dims[i - 1] * 2
            dim2 = dims[i]
            self.stages.append(
                TransformerStage(
                    img_size, dim1, dim2, depths[i],
                    stage_spec[i], groups[i], heads[i], strides[i],
                    attn_drop_rate, drop_rate, expansion, drop_rate,
                    dpr[sum(depths[:i]):sum(depths[:i + 1])],
                    ksizes[i], nat_ksizes[i])
                )
            img_size = img_size // 2

        self.down_projs = nn.ModuleList()
        for i in range(3):
            self.down_projs.append(
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1, bias=False),
                    LayerNormProxy(dims[i + 1])
                ) 
            )

        self.cls_norm = LayerNormProxy(dims[-1]) 
        self.cls_head = nn.Linear(dims[-1], num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def load_pretrained(self, state_dict, lookup_22k):

        new_state_dict = {}
        for state_key, state_value in state_dict.items():
            keys = state_key.split('.')
            m = self
            for key in keys:
                if key.isdigit():
                    m = m[int(key)]
                else:
                    m = getattr(m, key)
            if m.shape == state_value.shape:
                new_state_dict[state_key] = state_value
            else:
                # Ignore different shapes
                if 'relative_position_index' in keys:
                    new_state_dict[state_key] = m.data
                if 'q_grid' in keys:
                    new_state_dict[state_key] = m.data
                if 'reference' in keys:
                    new_state_dict[state_key] = m.data
                # Bicubic Interpolation
                if 'relative_position_bias_table' in keys:
                    n, c = state_value.size()
                    l_side = int(math.sqrt(n))
                    assert n == l_side ** 2
                    L = int(math.sqrt(m.shape[0]))
                    pre_interp = state_value.reshape(1, l_side, l_side, c).permute(0, 3, 1, 2)
                    post_interp = F.interpolate(pre_interp, (L, L), mode='bicubic')
                    new_state_dict[state_key] = post_interp.reshape(c, L ** 2).permute(1, 0)
                if 'rpe_table' in keys:
                    c, h, w = state_value.size()
                    C, H, W = m.data.size()
                    pre_interp = state_value.unsqueeze(0)
                    post_interp = F.interpolate(pre_interp, (H, W), mode='bicubic')
                    new_state_dict[state_key] = post_interp.squeeze(0)
                if 'cls_head' in keys:
                    new_state_dict[state_key] = state_value[lookup_22k]

        msg = self.load_state_dict(new_state_dict, strict=False)
        return msg

    def forward(self, x):
        x = self.patch_proj(x)
        for i in range(4):
            x = self.stages[i](x)
            if i < 3:
                x = self.down_projs[i](x)
        x = self.cls_norm(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.cls_head(x)
        return x, None, None


if __name__ == "__main__":
    model = DAT()
    test_data = torch.Tensor(1, 3, 224, 224)
    output = model(test_data)
    print(output.shape)
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""U-Nets for continuous-time Denoising Diffusion Probabilistic Models.

This file only serves as a helper for `examples/cont_ddpm.py`.

To use this file, run the following to install extra requirements:

pip install kornia
pip install einops
"""
import math

import kornia
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class Mish(nn.Module):
    def forward(self, x):
        return _mish(x)


@torch.jit.script
def _mish(x):
    return x * torch.tanh(F.softplus(x))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.register_buffer('emb', torch.exp(torch.arange(half_dim) * -emb))

    def forward(self, x):
        emb = x[:, None] * self.emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SelfAttention(nn.Module):
    def __init__(self, dim, groups=32, **kwargs):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, dim)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.group_norm(x)
        q, k, v = tuple(t.view(b, c, h * w) for t in self.qkv(x).chunk(chunks=3, dim=1))
        attn_matrix = (torch.bmm(k.permute(0, 2, 1), q) / math.sqrt(c)).softmax(dim=-2)
        out = torch.bmm(v, attn_matrix).view(b, c, h, w)
        return self.out(out)


class LinearTimeSelfAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, groups=32):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, dim)
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.group_norm(x)
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim, groups=8, dropout_rate=0.):
        super().__init__()
        # groups: Used in group norm.
        self.dim = dim
        self.dim_out = dim_out
        self.groups = groups
        self.dropout_rate = dropout_rate
        self.time_emb_dim = time_emb_dim

        self.mlp = nn.Sequential(
            Mish(),
            nn.Linear(time_emb_dim, dim_out)
        )
        # Norm -> non-linearity -> conv format follows
        # https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/nn.py#L55
        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Mish(),
            nn.Conv2d(dim, dim_out, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, dim_out),
            Mish(),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(dim_out, dim_out, 3, padding=1),
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1)

    def forward(self, x, t):
        h = self.block1(x)
        h += self.mlp(t)[..., None, None]
        h = self.block2(h)
        return h + self.res_conv(x)

    def __repr__(self):
        return (f"{self.__class__.__name__}(dim={self.dim}, dim_out={self.dim_out}, time_emb_dim="
                f"{self.time_emb_dim}, groups={self.groups}, dropout_rate={self.dropout_rate})")


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        f = f[None, None, :] * f[None, :, None]
        self.register_buffer('f', f)

    def forward(self, x):
        return kornia.filter2D(x, self.f, normalized=True)


class Downsample(nn.Module):
    def __init__(self, dim, blur=True):
        super().__init__()
        if blur:
            self.conv = nn.Sequential(
                Blur(),
                nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1),
            )
        else:
            self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, dim, blur=True):
        super().__init__()
        if blur:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1),
                Blur()
            )
        else:
            self.conv = nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self,
                 input_size=(3, 32, 32),
                 hidden_channels=64,
                 dim_mults=(1, 2, 4, 8),
                 groups=32,
                 heads=4,
                 dim_head=32,
                 dropout_rate=0.,
                 num_res_blocks=2,
                 attn_resolutions=(16,),
                 attention_cls=SelfAttention):
        super().__init__()
        in_channels, in_height, in_width = input_size
        dims = [hidden_channels, *map(lambda m: hidden_channels * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_pos_emb = SinusoidalPosEmb(hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 4),
            Mish(),
            nn.Linear(hidden_channels * 4, hidden_channels)
        )

        self.first_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)

        h, w = in_height, in_width
        self.down_res_blocks = nn.ModuleList([])
        self.down_attn_blocks = nn.ModuleList([])
        self.down_spatial_blocks = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            res_blocks = nn.ModuleList([
                ResnetBlock(
                    dim=dim_in,
                    dim_out=dim_out,
                    time_emb_dim=hidden_channels,
                    groups=groups,
                    dropout_rate=dropout_rate
                )
            ])
            res_blocks.extend([
                ResnetBlock(
                    dim=dim_out,
                    dim_out=dim_out,
                    time_emb_dim=hidden_channels,
                    groups=groups,
                    dropout_rate=dropout_rate
                ) for _ in range(num_res_blocks - 1)
            ])
            self.down_res_blocks.append(res_blocks)

            attn_blocks = nn.ModuleList([])
            if h in attn_resolutions and w in attn_resolutions:
                attn_blocks.extend(
                    [Residual(attention_cls(dim_out, heads=heads, dim_head=dim_head, groups=groups))
                     for _ in range(num_res_blocks)]
                )
            self.down_attn_blocks.append(attn_blocks)

            if ind < (len(in_out) - 1):
                spatial_blocks = nn.ModuleList([Downsample(dim_out)])
                h, w = h // 2, w // 2
            else:
                spatial_blocks = nn.ModuleList()
            self.down_spatial_blocks.append(spatial_blocks)

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(
            dim=mid_dim,
            dim_out=mid_dim,
            time_emb_dim=hidden_channels,
            groups=groups,
            dropout_rate=dropout_rate
        )
        self.mid_attn = Residual(attention_cls(mid_dim, heads=heads, dim_head=dim_head, groups=groups))
        self.mid_block2 = ResnetBlock(
            dim=mid_dim,
            dim_out=mid_dim,
            time_emb_dim=hidden_channels,
            groups=groups,
            dropout_rate=dropout_rate
        )

        self.ups_res_blocks = nn.ModuleList([])
        self.ups_attn_blocks = nn.ModuleList([])
        self.ups_spatial_blocks = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            res_blocks = nn.ModuleList([
                ResnetBlock(
                    dim=dim_out * 2,
                    dim_out=dim_out,
                    time_emb_dim=hidden_channels,
                    groups=groups,
                    dropout_rate=dropout_rate
                ) for _ in range(num_res_blocks)
            ])
            res_blocks.extend([
                ResnetBlock(
                    dim=dim_out + dim_in,
                    dim_out=dim_in,
                    time_emb_dim=hidden_channels,
                    groups=groups,
                    dropout_rate=dropout_rate
                )
            ])
            self.ups_res_blocks.append(res_blocks)

            attn_blocks = nn.ModuleList([])
            if h in attn_resolutions and w in attn_resolutions:
                attn_blocks.extend(
                    [Residual(attention_cls(dim_out, heads=heads, dim_head=dim_head, groups=groups))
                     for _ in range(num_res_blocks)]
                )
                attn_blocks.append(
                    Residual(attention_cls(dim_in, heads=heads, dim_head=dim_head, groups=groups))
                )
            self.ups_attn_blocks.append(attn_blocks)

            spatial_blocks = nn.ModuleList()
            if ind < (len(in_out) - 1):
                spatial_blocks.append(Upsample(dim_in))
                h, w = h * 2, w * 2
            self.ups_spatial_blocks.append(spatial_blocks)

        self.final_conv = nn.Sequential(
            nn.GroupNorm(groups, hidden_channels),
            Mish(),
            nn.Conv2d(hidden_channels, in_channels, 1)
        )

    def forward(self, t, x):
        t = self.mlp(self.time_pos_emb(t))

        hs = [self.first_conv(x)]
        for i, (res_blocks, attn_blocks, spatial_blocks) in enumerate(
                zip(self.down_res_blocks, self.down_attn_blocks, self.down_spatial_blocks)):
            if len(attn_blocks) > 0:
                for res_block, attn_block in zip(res_blocks, attn_blocks):
                    h = res_block(hs[-1], t)
                    h = attn_block(h)
                    hs.append(h)
            else:
                for res_block in res_blocks:
                    h = res_block(hs[-1], t)
                    hs.append(h)
            if len(spatial_blocks) > 0:
                spatial_block, = spatial_blocks
                hs.append(spatial_block(hs[-1]))

        h = hs[-1]
        h = self.mid_block1(h, t)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t)

        for i, (res_blocks, attn_blocks, spatial_blocks) in enumerate(
                zip(self.ups_res_blocks, self.ups_attn_blocks, self.ups_spatial_blocks)):
            if len(attn_blocks) > 0:
                for res_block, attn_block in zip(res_blocks, attn_blocks):
                    h = res_block(torch.cat((h, hs.pop()), dim=1), t)
                    h = attn_block(h)
            else:
                for res_block in res_blocks:
                    h = res_block(torch.cat((h, hs.pop()), dim=1), t)
            if len(spatial_blocks) > 0:
                spatial_block, = spatial_blocks
                h = spatial_block(h)
        return self.final_conv(h)

# -*- coding: utf-8 -*-
"""
@File : image_encoder.py
@Time : 2023/12/18 下午3:30
@Auth : Yue Zheng
@IDE  : Pycharm2022.3
@Ver. : Python3.9
@Comm : ···
"""
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    def __init__(
            self,
            image_size=1024,  # 输入图像尺寸大小
            patch_size=16,  # patch大小
            padding=0,  # padding的大小
            in_chans=3,  # 输入图像的channel
            embed_dim=768,  # 图像编码的channel
            depth=12,  # 主体编码器的深度
            num_heads=12,  # attention中head的个数
            mlp_ratio=4,  # mlp中channel缩放的比例
            out_chans=256,  # 输出特征的channel
            qkv_bias=True,  # qkv全连接层的偏置flag
            use_abs_pos=True,  # 是否使用相对位置嵌入添加到注意力图
            use_rel_pos=False,  # 是否使用绝对位置嵌入
            window_size=0,  # attention中窗口的大小
            global_attn_indexs=()  # 需要将rel_pos嵌入到注意力图的Encoder Block中
    ):
        super().__init__()
        self.image_size = image_size

        # patch embedding
        self.kernel_size = (patch_size, patch_size)  # 卷积核大小
        self.stride = (patch_size, patch_size)  # 步长
        self.padding = (padding, padding)  # padding
        self.chans = in_chans  # 输入channel
        self.embed_dim = embed_dim  # 输出channel
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding)  # 将每个图块投影到一个较高维度的特征空间中。

        # position embedding
        self.pos_embed = None
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        # 使用预训练图像大小(image_size//patch_size)初始化pos_embed
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, image_size // patch_size, image_size // patch_size, embed_dim)
            )

        # Transformer Encoder
        self.blocks = nn.ModuleList()
        # 由多个重复堆叠Encoder Block组成
        for i in range(depth):
            block = EncoderBlock(
                embed_dim,  # 输入的channel
                num_heads,  # attention中head的个数
                mlp_ratio,  # mlp中channel缩放的比例
                qkv_bias,  # qkv全连接层的偏置flag
                use_rel_pos,  # 是否要将rel_pos添加到注意力图
                window_size if i not in global_attn_indexs else 0,  # attention中窗口的大小
                (image_size // patch_size, image_size // patch_size))  # 输入特征的尺寸
            self.blocks.append(block)

        # Neck
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

    def forward(self, x):
        # patch embedding
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)

        # positional embedding
        # 添加位置编码
        if self.pos_embed is not None:
            x = x + self.pos_embed

        # Transformer encoder
        for b in self.blocks:
            x = b(x)

        # Neck
        # B C H W -> B H W C
        x = x.permute(0, 3, 1, 2)
        x = self.neck(x)
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)  # dim=1维度求均值并保留通道
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class EncoderBlock(nn.Module):
    def __init__(
            self,
            embed_dim,  # 输入的channel
            num_heads,  # attention中head的个数
            mlp_ratio=4,  # mlp中channel的缩放比
            qkv_bias=True,  # qkv全连阶层的偏置flag
            use_rel_pos=False,  # 是否要将rel_pos_embed添加到注意力图
            window_size=0,  # attention中的窗口大小
            input_size=()  # 输入特征的尺寸
    ):
        super().__init__()
        # Norm Layer
        self.norm1 = nn.LayerNorm(embed_dim)

        # Multi-Head Attention
        self.attn = MultiHeadAttn(
            embed_dim,
            num_heads,
            qkv_bias,
            use_rel_pos,
            input_size if window_size == 0 else (window_size, window_size)
        )

        # Norm Layer
        self.norm2 = nn.LayerNorm(embed_dim)

        # MLP
        self.mlp = MLPBlock(embed_dim, int(embed_dim * mlp_ratio))
        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)

        # 将输入序列划分为窗口，然后在每个窗口上执行注意力操作，以减少计算复杂性。
        # Window partition 对x进行padding
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        # 去除x的padding部分
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class MLPBlock(nn.Module):
    def __init__(self, embed_dim, mlp_dim):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, embed_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = nn.GELU()(x)
        x = self.linear2(x)
        return x


class MultiHeadAttn(nn.Module):
    def __init__(
            self,
            dim,  # 输入channel
            num_heads=8,  # head数目
            qkv_bias=True,
            use_rel_pos=False,
            input_size=None,  # 嵌入相对位置注意力特征的尺寸
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:  # 使用相对位置编码
            assert (
                    input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            # 2S-1,Epos
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv.shape =  (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # (q, k, v).shape = (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        # attn.shape = (B * nHead, H * W, H * W)
        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            # 假设use_rel_pos是True, (H, W) = (S, S)
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    # S, S
    q_h, q_w = q_size
    k_h, k_w = k_size
    # rel_pos_h -> 2S-1 × Epos
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    # torch.einsum用于简洁的表示乘积、点积、转置等方法
    # B, q_h, q_w, k_h
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    # B, q_h, q_w, k_w
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    # B,q_h, q_w, k_h, k_w
    attn = (
            attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


# get_rel_pos用于计算h和w的相对位置的嵌入特征
def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos. 相关位置进行插值
        rel_pos_resized = F.interpolate(
            # 1,N,Ep --> 1,Ep,N --> 1,Ep,2S-1
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        # Ep,2S-1 --> 2S-1,Ep
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    # 如果q和k长度值不同，则用短边长度缩放坐标。
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    # S, S
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    # tensor索引是tensor时,即tensor1[tensor2]
    # 假设tensor2某个具体位置值是2,则tensor1[2]位置的tensor1切片替换tensor2中的2
    # tensor1->shape 5,5,3 tensor2->shape 2,2,3 tensor1切片->shape 5,3 tensor1[tensor2]->shape 2,2,3,5,3
    # tensor1->shape 5,5 tensor2->shape 3,2,3 tensor1切片->shape 5 tensor1[tensor2]->shape 3,2,3,5

    # 2S-1, Ep --> S, S, Ep
    return rel_pos_resized[relative_coords.long()]


# Hp和Wp是S的整数倍
# window_partition调整了原始特征尺寸为(H×W–>S×S)，
# 目的是了在后续的Multi-Head Attention过程中将相对位置嵌入添加到注意力图(attn)，
# 并不是所有Block都需要在注意力图中嵌入相对位置信息。
# window_unpartition则是恢复特征的原始尺寸(S×S–>H×W)。
def window_partition(x, window_size):
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    # B, Hp/S, S, Wp/S, S, C
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    # B, Hp/S, Wp/S, S, S, C --> BHpWp/SS, S, S, C
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    # BHpWp/SS, S, S, C --> B, Hp/C, Wp/S, S, S, C
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    # B, Hp/S, Wp/S, S, S, C --> B, Hp/C, Wp/S, S, S, C
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()

    # B, H, W, C
    return x


if __name__ == '__main__':
    torch.manual_seed(42)
    img = torch.load("../a_tensor.pt")
    print(img.shape)
    vit = ImageEncoder(256, 16)
    preds = vit(img)
    print(preds)

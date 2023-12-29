# -*- coding: utf-8 -*-
"""
@File : prompt_encoder.py
@Time : 2023/12/18 下午7:47
@Auth : Yue Zheng
@IDE  : Pycharm2022.3
@Ver. : Python3.9
@Comm : ···
"""
import numpy as np
import torch
from torch import nn


class PromptEncoder(nn.Module):
    def __init__(
            self,
            embed_dim,  # prompt encoder的channel
            image_size,  # 输入image的size
            image_embed_size,  # mask的编码尺寸
            mask_in_chans  # mask编码的channel
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = image_size
        self.image_embed_size = image_embed_size  # mask的编码尺寸
        self.mask_in_chans = mask_in_chans  # mask编码的channel
        self.pe_layer = PosEmbedRandom(embed_dim // 2)

        self.num_point_embed = 4  # 4个点:正负点,框的俩个点
        # 4个点的embed向量
        point_embed = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embed)]
        # nn.ModuleList存储不同的module，自动将每个module的parameters添加到网络之中
        self.point_embed = nn.ModuleList(point_embed)  # 将4个点的embed向量添加到神经网络
        self.not_a_point_embed = nn.Embedding(1, embed_dim)  # 不是点的嵌入向量
        self.mask_input_size = (4 * image_embed_size[0], 4 * image_embed_size[1])  # mask的输入尺寸
        # 输入mask时，进行4倍下采样，保持mask输出尺寸和image encoder输出尺寸保持一致
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        # 生成没有 mask 输入时的嵌入向量
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def forward(self, points=None, boxes=None, masks=None):
        # 获得batch_size，当前的predict为1
        batch_size = self.get_batch_size(points, boxes, masks)

        # sparse embedding
        device = self.point_embed[0].weight.device  # 获取设备型号
        sparse_embed = torch.empty((batch_size, 0, self.embed_dim),
                                   device=device)

        # 标记点编码，标记点points -> 向量points_embed
        if points is not None:
            coords, labels = points
            points_embed = self.embed_points(coords, labels, pad=(boxes is None))
            sparse_embed = torch.cat([sparse_embed, points_embed], dim=1)

        # 标记框编码，标记框boxes -> 向量boxes_embed
        if boxes is not None:
            box_embed = self.embed_boxes(boxes)
            sparse_embed = torch.cat([sparse_embed, box_embed], dim=1)

        # dense embedding
        if masks is not None:
            # mask编码，mask下采样保证和image encoder输出一致
            dense_embed = self.embed_masks(masks)
        else:
            # 假如没有mask输入，则将no_mask_embed编码扩展到与image encoder一致的尺寸代替mask
            dense_embed = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                batch_size, -1, self.image_embed_size[0], self.image_embed_size[1]
            )
        return sparse_embed, dense_embed

    # 标记点预处理，将channel由2变成embed_dim(MatMul: forward_with_coords)，然后再加上位置编码权重。
    # 2：坐标(h, w) --> embed_dim：提示编码的channel
    def embed_points(self, points, labels, pad):
        points = points + 0.5  # 移动到像素中心
        # points和boxes联合则不需要pad，pad的作用相当于box占位符号。
        # box和points可以联合标定完成图像分割的，但是此时的box只能有一个，不能有多个。
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)  # B, 1, 2
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)  # B, 1
            points = torch.cat([points, padding_point], dim=1)  # B,N+1,2
            labels = torch.cat([labels, padding_label], dim=1)  # B,N+1

        points_embed = self.pe_layer.forward_with_coords(points, self.input_image_size)  # B, N+1, 2f
        # labels=-1是非标记点，设为非标记点权重
        points_embed[labels == -1] = 0
        points_embed[labels == -1] += self.not_a_point_embed.weight
        # labels=0是背景点，加上背景点权重
        points_embed[labels == 0] += self.point_embed[0].weight
        # labels=1是目标点，加上目标点权重
        points_embed[labels == 1] += self.point_embed[1].weight
        return points_embed

    # 标记框预处理，将channel由4到2再变成embed_dim(MatMul: forward_with_coords)，然后再加上位置编码权重。
    # 4：起始点与末位点坐标(h1, w1, h2, w2) --> 2：坐标(h, w) --> embed_dim：提示编码的channel
    # boxes reshape 后 batchsize是会增加的，B, N, 4 –-> BN, 2, 2
    # 因此这里可以得出box和points联合标定时，box为什么只能是一个，而不能是多个。
    def embed_boxes(self, boxes):
        boxes = boxes + 0.5  # 移动到像素中心
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        # 目标框起始点和末尾点分别加上权重
        corner_embedding[:, 0, :] += self.point_embed[2].weight
        corner_embedding[:, 1, :] += self.point_embed[3].weight
        return corner_embedding

    # mask的输出尺寸是image encoder模块输出的图像编码尺寸的4倍，因此为了保持一致，需要4倍下采样。
    def embed_masks(self, masks):
        # mask下采样4倍数
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    # 获得batch size
    def get_batch_size(self, points=None, boxes=None, masks=None):
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    # # 获得设备型号
    # def _get_device(self):
    #     return self.point_embed[0].weight.device

    def get_dense_pe(self):
        # 返回用于对点提示进行编码的位置编码，将image encoder的形状应用于密集的点集。
        # 1x(embed_dim)x(embedding_h)x(embedding_w)
        return self.pe_layer(self.image_embed_size).unsqueeze(0)


# 用于将points和boxes的坐标进行prompt encoder
class PosEmbedRandom(nn.Module):
    def __init__(self, num_pos_feat=64, scale=None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        # 这个模型的positional_encoding_gaussian_matrix常数，[2, f]
        self.register_buffer("positional_encoding_gaussian_matrix",
                             scale * torch.randn((2, num_pos_feat)))

    # 通过sin和cos将编码的值归一化到[-1,1]
    def pos_emb_encoding(self, coords):
        coords = 2 * coords - 1
        # B, N+1, 2 * 2, f --> B, N+1, f
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # B, N+1, 2f
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size):
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self.pos_emb_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)

    # 将标记点的坐标具体的位置转变为[0~1]之间的比例位置
    def forward_with_coords(self, coords_input, image_size):
        coords = coords_input.clone()
        # 将坐标位置缩放到[0-1]之间
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        # B, N+1, 2 --> B, N+1, 2f
        return self.pos_emb_encoding(coords.to(torch.float))


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


if __name__ == '__main__':
    torch.manual_seed(42)
    # 创建一个 PromptEncoder 实例
    embed_dim = 256
    image_embedding_size = (14, 14)
    input_image_size = (224, 224)
    mask_in_chans = 16

    prompt_encoder = PromptEncoder(
        embed_dim=embed_dim,
        image_embed_size=image_embedding_size,
        image_size=input_image_size,
        mask_in_chans=mask_in_chans
    )

    # 创建一些测试数据
    points = (torch.randn(3, 2, 2), torch.randint(0, 2, (3, 2)))
    # boxes = None
    boxes = torch.randn(3, 4)
    # masks = None
    masks = torch.randn(4, 1, 28, 28)

    # points = None
    # masks = None
    # 直接复用上面创建的实例
    # 通过数据测试forward函数
    embeddings = prompt_encoder(points, boxes, masks)

    # 打印outout的shape
    # print(embeddings[0].shape)
    # print(embeddings[1].shape)
    print(embeddings)
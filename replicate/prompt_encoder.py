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
    def __init__(self, embed_dim, image_embed_size, input_img_size, mask_in_chans):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embed_size = image_embed_size
        self.input_img_size = input_img_size
        self.mask_in_chans = mask_in_chans
        self.pe_layer = PosEmbedRandom(embed_dim // 2)

        self.num_point_embed = 4
        point_embed = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embed)]
        self.point_embed = nn.ModuleList(point_embed)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embed_size[0], 4 * image_embed_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def forward(self, points=None, boxes=None, masks=None):
        batch_size = self._get_batch_size(points, boxes, masks)
        sparse_embed = torch.empty((batch_size, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords, labels = points
            points_embed = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embed = torch.cat([sparse_embed, points_embed], dim=1)
        if boxes is not None:
            box_embed = self._embed_boxes(boxes)
            sparse_embed = torch.cat([sparse_embed, box_embed], dim=1)
        if masks is not None:
            dense_embed = self._embed_masks(masks)
        else:
            dense_embed = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                batch_size, -1, self.image_embed_size[0], self.image_embed_size[1]
            )
        return sparse_embed, dense_embed

    def _embed_points(self, points, labels, pad):
        "_embed_points: 将输入的点坐标(points)和标签(labels)嵌入到一个稠密的向量表示中"
        points = points + 0.5
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        points_embed = self.pe_layer.forward_with_coords(points, self.input_img_size)
        points_embed[labels == -1] = 0
        points_embed[labels == -1] += self.not_a_point_embed.weight
        points_embed[labels == 0] += self.point_embed[0].weight
        points_embed[labels == 1] += self.point_embed[1].weight
        return points_embed

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_img_size)
        corner_embedding[:, 0, :] += self.point_embed[2].weight
        corner_embedding[:, 1, :] += self.point_embed[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(self, points=None, boxes=None, masks=None):
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self):
        return self.point_embed[0].weight.device

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)


class PosEmbedRandom(nn.Module):
    def __init__(self, num_pos_feat=64, scale=None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer("positional_encoding_gaussian_matrix",
                             scale * torch.randn((2, num_pos_feat)))

    def _pe_encoding(self, coords):
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size):
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)

    def forward_with_coords(self, coords_input, image_size):
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))


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
    # 创建一个 PromptEncoder 实例
    embed_dim = 256
    image_embedding_size = (14, 14)
    input_image_size = (224, 224)
    mask_in_chans = 16

    prompt_encoder = PromptEncoder(
        embed_dim=embed_dim,
        image_embed_size=image_embedding_size,
        input_img_size=input_image_size,
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
    print(embeddings[0].shape)
    print(embeddings[1].shape)

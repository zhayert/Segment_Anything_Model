# -*- coding: utf-8 -*-
"""
@File : mask_decoder.py
@Time : 2023/12/19 下午7:30
@Auth : Yue Zheng
@IDE  : Pycharm2022.3
@Ver. : Python3.9
@Comm : ···
"""
import torch
from torch import nn
from torch.nn import functional as F

from segment_anything.modeling import TwoWayTransformer


# from typing import

class MaskDecoder(nn.Module):
    def __init__(self, trm_dim, trm, num_multitask_outputs=3, iou_head_dpt=3, iou_head_hidden_dim=256):
        super().__init__()
        self.trm_dim = trm_dim
        self.trm = trm

        self.num_multitask_outputs = num_multitask_outputs

        self.iou_token = nn.Embedding(1, trm_dim)
        self.num_mask_tokens = num_multitask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, trm_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(trm_dim, trm_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(trm_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(trm_dim // 4, trm_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )
        self.output_hypernetworks_mlp = nn.ModuleList(
            [
                MLP(trm_dim, trm_dim, trm_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ])
        self.iou_pred_head = MLP(trm_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_dpt)

    def forward(self, img_emb, img_pe, sparse_prompt_embed, dense_prompt_embed, multimask_output=False):
        masks, iou_pred = self.pred_mask(img_emb, img_pe, sparse_prompt_embed, dense_prompt_embed)

        # select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred

    def pred_mask(self, img_embed, img_pe, sparse_prompt_embed, dense_prompt_embed):
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        # 5,E --> B,5,E
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embed.size(0), -1, -1)
        # concat(B,5,E and B,N,E) --> B,5+N,E       N是点的个数(标记点和标记框的点)
        tokens = torch.cat((output_tokens, sparse_prompt_embed), dim=1)

        # B, C, H, W
        src = torch.repeat_interleave(img_embed, tokens.shape[0], dim=0)
        # B, C, H, W + 1, C, H, W --> B, C, H, W
        src = src + dense_prompt_embed
        # 1, C, H, W --> B, C, H, W
        pos_src = torch.repeat_interleave(img_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # transformer
        hs, src = self.trm(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]

        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embed = self.output_upscaling(src)
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlp[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embed.shape
        masks = (hyper_in @ upscaled_embed.view(b, c, h * w).view(b, -1, h, w))

        # Generate mask equality predictions
        iou_pred = self.iou_pred_head(iou_token_out)
        return masks, iou_pred


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, sigmoid_output=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < self.num_layers - 1:
                x = F.relu(layer(x))
            else:
                x = layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def test1():
    # 创建一个包含两个整数的张量
    a = torch.tensor([0, 4])

    # 实例化嵌入层
    iou_token = nn.Embedding(5, 123)  # 这里使用6是因为可能的整数索引为0到5

    # 使用嵌入层获取嵌入向量
    embedding_result = iou_token(a)

    print(a.shape)
    print(embedding_result.shape)


def test2():
    # 定义 MLP 模型的参数
    input_dim = 10
    hidden_dim = 20
    output_dim = 1
    num_layers = 3
    sigmoid_output = True

    # 创建 MLP 模型实例
    model = MLP(input_dim, hidden_dim, output_dim, num_layers, sigmoid_output)

    # 生成随机输入数据
    input_data = torch.randn(5, input_dim)  # 5 个样本，每个样本有 10 个特征

    # 调用前向传播方法
    output = model(input_data)

    # 打印输出结果
    print("Input Data:")
    print(input_data)
    print("\nOutput of MLP:")
    print(output)



if __name__ == '__main__':
    transformer_dim = 256
    transformer = TwoWayTransformer(
        depth=2,
        embedding_dim=256,
        mlp_dim=2048,
        num_heads=8,
    )  # 需要提前定义
    image_size = (14, 14)

    # 构建Model
    model = MaskDecoder(
        trm_dim=transformer_dim,
        trm=transformer
    )

    # 测试数据
    image_embeddings = torch.rand(2, transformer_dim, 14, 14)
    image_pe = torch.rand(2, 256, 196)
    sparse_prompt_embeddings = torch.rand(2, 3, transformer_dim)
    dense_prompt_embeddings = torch.rand(4, transformer_dim, 14, 14)

    # 测试前向传播
    masks, iou_pred = model(
        img_emb=image_embeddings,
        img_pe=image_pe,
        sparse_prompt_embed=sparse_prompt_embeddings,
        dense_prompt_embed=dense_prompt_embeddings,
        multimask_output=False
    )

    # 打印输出大小
    print(masks.shape)
    print(iou_pred.shape)

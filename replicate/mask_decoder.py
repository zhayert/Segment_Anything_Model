# -*- coding: utf-8 -*-
"""
@File : mask_decoder.py
@Time : 2023/12/19 下午7:30
@Auth : Yue Zheng
@IDE  : Pycharm2022.3
@Ver. : Python3.9
@Comm : ···
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


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


# if __name__ == '__main__':
#     transformer_dim = 256
#     transformer = TwoWayTransformer(
#         depth=2,
#         embedding_dim=256,
#         mlp_dim=2048,
#         num_heads=8,
#     )  # 需要提前定义
#     image_size = (14, 14)
#
#     # 构建Model
#     model = MaskDecoder(
#         trm_dim=transformer_dim,
#         trm=transformer
#     )
#
#     # 测试数据
#     image_embeddings = torch.rand(2, transformer_dim, 14, 14)
#     image_pe = torch.rand(2, 256, 196)
#     sparse_prompt_embeddings = torch.rand(2, 3, transformer_dim)
#     dense_prompt_embeddings = torch.rand(4, transformer_dim, 14, 14)
#
#     # 测试前向传播
#     masks, iou_pred = model(
#         img_emb=image_embeddings,
#         img_pe=image_pe,
#         sparse_prompt_embed=sparse_prompt_embeddings,
#         dense_prompt_embed=dense_prompt_embeddings,
#         multimask_output=False
#     )
#
#     # 打印输出大小
#     print(masks.shape)
#     print(iou_pred.shape)


# 双向Transformer结构
class TwoWayTransformer(nn.Module):
    def __init__(self, depth, embed_dim, num_heads, mlp_dim, attention_downsample_rate=2):
        super().__init__()
        self.depth = depth
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        # 包含多个TransformerBlock结构每个块都有两个方向的注意力机制。
        # 这些块被堆叠在一起，通过残差连接形成了一个深层的 Transformer 结构。
        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        # 在整个Transformer结构的最后，应用了一个额外的注意力层，保持性能的同时，提高模型的计算效率。
        self.final_attn_token_to_image = Attention(
            embed_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embed_dim)

    def forward(self, image_embed, image_pe, point_embed):
        # 形状变换：在处理图像嵌入时，首先对其进行了形状变换，将其从形状为 BxCxHxW 的四维张量
        # 展平为形状为 Bx(N_image_tokens)xC 的三维张量，以便在 Transformer 中使用。
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embed.shape
        image_embed = image_embed.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        queries = point_embed
        keys = image_embed

        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embed,
                key_pe=image_pe,
            )

        q = queries + point_embed
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q, k, v)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 mlp_dim,
                 attention_downsample_rate=2,
                 skip_first_layer_pe=False):
        super().__init__()
        self.self_attn = Attention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.cross_attn_token_to_image = Attention(
            embed_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        self.lin1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.lin2 = nn.Linear(mlp_dim, embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.norm4 = nn.LayerNorm(embed_dim)
        self.cross_attn_token_to_image = Attention(embed_dim, num_heads, attention_downsample_rate)

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries, keys, query_pe, key_pe):
        # 在这个函数中，query_pe和key_pe分别表示queries和keys的positional embeddings。
        # 在Transformer模型中，位置嵌入用于为序列中的每个元素赋予一个唯一的位置编码，帮助模型理解元素之间的相对位置关系。
        if self.skip_first_layer_pe:
            queries = self.self_attn(queries, queries, queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q, q, queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # 使模型中tokens加入到image embedding
        # 在处理序列数据时，有效地利用图像的信息，提高对序列中每个令牌的建模能力。
        # 将输入的queries和query_pe相加。这一步的目的是引入关于查询令牌位置的信息。
        q = queries + query_pe
        k = keys + key_pe
        # 使用cross_attn_token_to_image注意力层，计算注意力加权的输出attn_out。
        attn_out = self.cross_attn_token_to_image(q, k, keys)
        # 将原始的queries与attn_out相加，得到更新后的queries。
        queries = queries + attn_out
        queries = self.norm2(queries)

        # 将更新后的queries输入到多层感知机块mlp中，得到mlp_out。
        # 将原始的queries与mlp_out相加，得到更新后的queries。
        # 对更新后的queries进行层归一化，得到最终的输出。
        mlp_out = self.lin1(queries)
        mlp_out = self.act(mlp_out)
        mlp_out = self.lin2(mlp_out)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # 再次的跨通道注意力块（CrossAttentionBlock）
        # 将queries与查询的位置嵌入query_pe相加，将keys与键的位置嵌入key_pe相加。
        # 使用cross_attn_image_to_token注意力层，计算注意力加权的输出attn_out。
        # 将原始的keys与attn_out相加，得到更新后的keys。对更新后的keys进行层归一化，得到最终的输出。
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(k, q, queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, downsample_rate=1):
        super().__init__()
        self.embed_dim = embed_dim
        # internal_dim 表示在进行注意力计算时，嵌入向量会被投影到一个更低维度的空间。
        # 以减少计算的开销，同时保持模型的表达能力。
        self.internal_dim = embed_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_head must divide embed_dim."

        self.q_proj = nn.Linear(embed_dim, self.internal_dim)
        self.k_proj = nn.Linear(embed_dim, self.internal_dim)
        self.v_proj = nn.Linear(embed_dim, self.internal_dim)

        self.out_proj = nn.Linear(self.internal_dim, embed_dim)

    def separate_heads(self, x, num_heads):
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        x = x.transpose(1, 2)
        return x

    def forward(self, q, k, v):
        # 投影
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # 多头分离
        q = self.separate_heads(q, self.num_heads)
        k = self.separate_heads(k, self.num_heads)
        v = self.separate_heads(v, self.num_heads)

        # 自注意力机制
        _, _, _, c_per_head = q.shape  # c_per_head
        attn = q @ k.permute(0, 1, 3, 2)  # Batch_size x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # 输出计算
        out = attn @ v

        # 用于将多头注意力计算的输出重新组合成一个张量,为了将多个头的注意力计算结果合并
        b, num_heads, num_tokens, c_per_head = out.shape
        out = out.transpose(1, 2)
        out = out.reshape(b, num_heads, num_tokens * c_per_head)

        return out

# class MLPBlock(nn.Module):
#     def __init__(self,embed_dim,mlp_dim):
#         super().__init__()
#         self.lin1 = nn.Linear(embed_dim,mlp_dim)
#         self.lin2 = nn.Linear(mlp_dim,embed_dim)
#
#     def forward(self,x):
#         x = self.lin1(x)
#         x = nn.GELU()(x)
#         x = self.lin2(x)
#         return x

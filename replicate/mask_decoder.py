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


class MaskDecoder(nn.Module):
    def __init__(
            self,
            trm_dim,  # tranformer的channel
            trm,  # 用于预测mask的网络transformer
            num_multitask_outputs=3,  # 消除掩码歧义预测的掩码数
            iou_head_dpt=3,  # MLP深度，MLP用于预测mask的质量
            iou_head_hidden_dim=256  # MLP隐藏的channel
    ):
        super().__init__()
        self.trm_dim = trm_dim  # transformer的channel

        # transformer：融合特征(提示信息特征与图像特征)获得粗略掩膜src
        self.trm = trm  # 用于预测mask的网络transformer

        self.num_multitask_outputs = num_multitask_outputs  # 消除mask歧义预测的掩码数
        self.iou_token = nn.Embedding(1, trm_dim)  # iou的token
        self.num_mask_tokens = num_multitask_outputs + 1  # mask数
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, trm_dim)  # mask的tokens数

        # upscaled：对粗略掩膜src进行4倍上采样
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(trm_dim, trm_dim // 4, kernel_size=2, stride=2),  # 转置卷积 上采样2倍
            LayerNorm2d(trm_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(trm_dim // 4, trm_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )

        # MLP
        # 对应mask数的MLP：全连接层组(计算加权权重，使粗掩膜src转变为掩膜mask)
        self.output_hypernetworks_mlp = nn.ModuleList(
            [
                MLPBlock(trm_dim, trm_dim, trm_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ])

        # 对应iou的MLP：全连接层组(计算掩膜mask的Score)
        self.iou_pred_head = MLPBlock(
            trm_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_dpt)

    def forward(
            self,
            image_embed,  # image encoder图像特征
            image_pos_embed,  # image的pos embed
            sparse_prompt_embed,  # 标记点和标记框的embed
            dense_prompt_embed,  # 输入mask的embed
            multimask_output=False  # 是否输出多个mask
    ):

        masks, iou_pred = self.pred_mask(image_embed, image_pos_embed, sparse_prompt_embed, dense_prompt_embed)

        # 选择正确的一个或多个mask进行输出
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred

    def pred_mask(
            self,
            image_embed,
            image_pos_embed,
            sparse_prompt_embed,
            dense_prompt_embed):
        # 拼接output_tokens
        # 1,E and 4,E --> 5,E
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        # 5,E --> B,5,E
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embed.size(0), -1, -1)
        # concat(B,5,E and B,N,E) --> B,5+N,E       N是点的个数(标记点和标记框的点)
        tokens = torch.cat((output_tokens, sparse_prompt_embed), dim=1)

        # 扩展image_embedd的B维度,因为boxes标记分割时,n个box时batchsize = batchsize * n
        # B, C, H, W
        src = torch.repeat_interleave(image_embed, tokens.shape[0], dim=0)
        # B, C, H, W + 1, C, H, W --> B, C, H, W
        src = src + dense_prompt_embed
        # 1, C, H, W --> B, C, H, W
        pos_src = torch.repeat_interleave(image_pos_embed, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # transformer
        # Run the transformer
        # B, N, C
        hs, src = self.trm(src, pos_src, tokens)

        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]

        # 对mask_embed(src)进行上采样，并且使用mask_tokens来预测masks掩码
        # B,N,C-->B,C,H,W
        src = src.transpose(1, 2).view(b, c, h, w)

        # upscaled 4倍上采样
        # 在MaskDecoder的predict_masks添加位置编码
        upscaled_embed = self.output_upscaling(src)

        hyper_in_list = []

        # MLP Block
        # 在MaskDecoder的predict_masks添加位置编码
        for i in range(self.num_mask_tokens):
            # mask_tokens_out[:, i, :]: B,1,C
            # output_hypernetworks_mlps: B,1,c
            hyper_in_list.append(self.output_hypernetworks_mlp[i](mask_tokens_out[:, i, :]))

        # B, n, c
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embed.shape
        # B,n,c × B,c,N-->B,n,h,w
        masks = (hyper_in @ upscaled_embed.view(b, c, h * w)).view(b, -1, h, w)

        # --- iou MLP ---
        # Generate mask quality predictions
        # iou_token_out: B,1,n
        iou_pred = self.iou_pred_head(iou_token_out)
        # --- iou MLP ---

        # mask: B, n, h, w
        # iou_pred: B, 1, n
        return masks, iou_pred


class MLPBlock(nn.Module):
    def __init__(
            self,
            input_dim,  # 输入channel
            hidden_dim,  # 中间channel
            output_dim,  # 输出channel
            num_layers,  # fc的层数
            sigmoid_output=False
    ):
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
    model = MLPBlock(input_dim, hidden_dim, output_dim, num_layers, sigmoid_output)

    # 生成随机输入数据
    input_data = torch.randn(5, input_dim)  # 5 个样本，每个样本有 10 个特征

    # 调用前向传播方法
    output = model(input_data)

    # 打印输出结果
    print("Input Data:")
    print(input_data)
    print("\nOutput of MLP:")
    print(output)


# 双向Transformer结构
class TwoWayTransformer(nn.Module):
    def __init__(
            self,
            depth,  # 层数
            embed_dim,  # 输入的channel
            num_heads,  # attention的head数
            mlp_dim,  # MLP内部的channel
            # 用于对queries和keys的下采样，这可以在注意力机制中引入空间平均池化，以降低计算复杂性。
            # 这种方式下采样旨在在保持有效性的同时减少计算需求。
            attention_downsample_rate=2
    ):
        super().__init__()
        self.depth = depth  # 层数
        self.embed_dim = embed_dim  # 输入的channel
        self.num_heads = num_heads  # attention的head数
        self.mlp_dim = mlp_dim  # MLP内部隐藏的channel
        self.layers = nn.ModuleList()

        # 包含多个TransformerBlock结构每个块都有两个方向的注意力机制。
        # 这些块被堆叠在一起，通过残差连接形成了一个深层的 Transformer 结构。
        for i in range(depth):
            self.layers.append(
                TwoWayAttnBlock(
                    embed_dim=embed_dim,  # 输入channel
                    num_heads=num_heads,  # attention的head数
                    mlp_dim=mlp_dim,  # MLP中间channel
                    attention_downsample_rate=attention_downsample_rate,  # 下采样
                    skip_first_layer_pe=(i == 0),
                )
            )

        # 在整个Transformer结构的最后，应用了一个额外的注意力层，保持性能的同时，提高模型的计算效率。
        self.final_attn_token_to_image = MultiHeadAttn(
            embed_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embed_dim)

    def forward(self, image_embed, image_pos_embed, point_embed):
        # 进行了形状变换， BxCxHxW -> Bx(N_image_tokens)xC,以便在 Transformer 中使用。
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embed.shape

        # image_embed
        # BxHWxC => B,N,C
        image_embed = image_embed.flatten(2).permute(0, 2, 1)

        # image_pos_embed
        # BxHWxC => B,N,C
        image_pos_embed = image_pos_embed.flatten(2).permute(0, 2, 1)

        # 标记点编码
        # B，N，C
        queries = point_embed
        keys = image_embed

        # --- TwoWayAttention---
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pos_embed=point_embed,
                key_pos_embed=image_pos_embed,
            )
        # --- TwoWayAttention---

        q = queries + point_embed
        k = keys + image_pos_embed

        # --- Attention ---
        attn_out = self.final_attn_token_to_image(q, k, v=keys)
        # --- Attention ---

        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


# TwoWayAttention Block由LayerNorm, Multi-Head, Attention和MLP构成。
# TwoWayAttentionBlock是Prompt encoder的提示信息特征与Image encoder的图像特征的融合过程，
class TwoWayAttnBlock(nn.Module):
    def __init__(
            self,
            embed_dim,  # 输入的channel
            num_heads,  # attention的head
            mlp_dim,  # MLP中的channel
            attention_downsample_rate=2,  # 下采样
            skip_first_layer_pe=False
    ):
        super().__init__()
        self.self_attn = MultiHeadAttn(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.cross_attn_token_to_image = MultiHeadAttn(
            embed_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        self.lin1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.ReLU()
        self.lin2 = nn.Linear(mlp_dim, embed_dim)

        self.norm3 = nn.LayerNorm(embed_dim)

        self.norm4 = nn.LayerNorm(embed_dim)

        self.cross_attn_image_to_token = MultiHeadAttn(
            embed_dim, num_heads, attention_downsample_rate)

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
            self,
            queries,  # 标记点编码相关（原始标记点编码经过一系列特征提取） queries
            keys,  # 原始图像编码相关（原始图像编码经过一些列特征提取）  keyes
            query_pos_embed,  # 原始标记点编码  point_embed
            key_pos_embed  # 原始图像位置编码  image_pos_embed
    ):
        if self.skip_first_layer_pe:
            # 第一轮本身queries == query_pos_embed没法在比较残差
            queries = self.self_attn(queries, queries, queries)
        else:
            q = queries + query_pos_embed
            attn_out = self.self_attn(q, q, queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pos_embed
        k = keys + key_pos_embed
        attn_out = self.cross_attn_token_to_image(q, k, keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP Block
        mlp_out = self.lin1(queries)
        mlp_out = self.act(mlp_out)
        mlp_out = self.lin2(mlp_out)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attention to tokens
        q = queries + query_pos_embed
        k = keys + key_pos_embed
        attn_out = self.cross_attn_image_to_token(k, q, queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        return queries, keys


# Mask Decoder的Attention与ViT的Attention有些细微的不同：
# Mask Decoder的Attention是3个FC层分别接受3个输入获得q、k和v。
# ViT的Attention是1个FC层接受1个输入后将结果均拆分获得q、k和v。
class MultiHeadAttn(nn.Module):
    def __init__(
            self,
            embed_dim,  # 输入channel
            num_heads,  # attention的head数
            downsample_rate=1  # 下采样
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # internal_dim 表示在进行注意力计算时，嵌入向量会被投影到一个更低维度的空间。
        # 以减少计算的开销，同时保持模型的表达能力。
        self.internal_dim = embed_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_head must divide embed_dim."

        # qkv获取
        self.q_proj = nn.Linear(embed_dim, self.internal_dim)
        self.k_proj = nn.Linear(embed_dim, self.internal_dim)
        self.v_proj = nn.Linear(embed_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embed_dim)

    def separate_heads(self, x, num_heads):
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        x = x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head
        return x

    def forward(self, q, k, v):
        # 输入投影
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # 多头分离
        # B,N_heads,N_tokens,C_per_head
        q = self.separate_heads(q, self.num_heads)
        k = self.separate_heads(k, self.num_heads)
        v = self.separate_heads(v, self.num_heads)

        # Attention自注意力机制
        _, _, _, c_per_head = q.shape  # c_per_head
        attn = q @ k.permute(0, 1, 3, 2)  # Batch_size x N_heads x N_tokens x N_tokens
        # Scale
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        # 输出计算
        out = attn @ v

        # 用于将多头注意力计算的输出重新组合成一个张量,为了将多个头的注意力计算结果合并
        b, num_heads, num_tokens, c_per_head = out.shape
        out = out.transpose(1, 2)
        out = out.reshape(b, num_tokens, num_heads * c_per_head)  # B x N_tokens x C

        out = self.out_proj(out)
        return out


if __name__ == '__main__':
    # 构建TwoWayTransformer
    transformer_dim = 256
    transformer = TwoWayTransformer(depth=2, embed_dim=256, mlp_dim=2048, num_heads=8)

    # 构建MaskDecoder
    model = MaskDecoder(trm_dim=transformer_dim, trm=transformer)

    # 生成匹配的输入
    bs = 2
    seq_len = 3
    image_size = 14
    transformer_dim = 256

    image_embeddings = torch.rand(bs, transformer_dim, image_size, image_size)
    image_pe = torch.rand(bs, transformer_dim, image_size ** 2)
    sparse_prompt_embeddings = torch.rand(bs, seq_len, transformer_dim)

    # 修正dense_prompt_embeddings的shape
    dense_prompt_embeddings = torch.rand(bs, transformer_dim, image_size, image_size)

    masks, iou_pred = model(
        image_embed=image_embeddings,
        image_pos_embed=image_pe,
        sparse_prompt_embed=sparse_prompt_embeddings,
        dense_prompt_embed=dense_prompt_embeddings,
        multimask_output=False
    )

    print(masks.shape)
    print(iou_pred.shape)

# -*- coding: utf-8 -*-
"""
@File : sam.py
@Time : 2023/12/20 下午8:35
@Auth : Yue Zheng
@IDE  : Pycharm2022.3
@Ver. : Python3.9
@Comm : ···
"""

import torch
from torch import nn


class Sam(nn.Module):
    mask_shreshold = 0.0
    image_format = "RGB"

    def __init__(self, image_encoder, prompt_encoder, mask_decoder,
                 pixel_mean=[123.675, 116.28, 103.53],
                 pixel_std=[58.395, 57.12, 68.375]):
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self):
        return self.pixel_mean.device

    @torch.no_grad
    def forward(self, batch_input, multimask_output):
        input_images = torch.stack([self.preprocess(x["image"]) for x in batch_input],dim = 0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        # for image_record, cirr



def forward(self):
    return


def build_sam_vit_b(checkpoint=None,
                    encoder_emb_dim=768,
                    encoder_depth=12,
                    encoder_num_heads=12,
                    encoder_global_attn_idx=[2, 5, 8, 11]):
    prompt_emb_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_emb_size = image_size // vit_patch_size
    # sam

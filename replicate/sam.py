# -*- coding: utf-8 -*-
"""
@File : sam.py
@Time : 2023/12/20 下午8:35
@Auth : Yue Zheng
@IDE  : Pycharm2022.3
@Ver. : Python3.9
@Comm : ···
"""
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torchvision.transforms.functional import resize, to_pil_image

from copy import deepcopy

from replicate.image_encoder import ImageEncoder
from replicate.mask_decoder import MaskDecoder, TwoWayTransformer
from replicate.prompt_encoder import PromptEncoder


class Sam(nn.Module):
    mask_shreshold = 0.0
    image_format = "RGB"

    def __init__(self, image_encoder: ImageEncoder,
                 prompt_encoder: PromptEncoder,
                 mask_decoder: MaskDecoder,
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
        # You can try it
        for i in batch_input:
            x = (i - self.pixel_mean) / self.pixel_std
            h, w = x.shape[-2:]
            padh = self.image_encoder.image_size - h
            padw = self.image_encoder.image_size - w
            x = F.pad(x, (0, padw, 0, padh))

            input_images = torch.stack(x, dim=0)

        # input_images = torch.stack([self.preprocess(x["image"]) for x in batch_input], dim=0)
        image_embed = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embed in zip(batch_input, image_embed):
            if "point_corrds" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embed, dense_embed = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None)
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embed=curr_embed.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_embed=sparse_embed,
                dense_embed=dense_embed,
                multimask_output=multimask_output,
            )
            # 使用F.interpolate函数将输入的掩码masks放缩到一个固定大小，
            # 这个大小由self.image_encoder.img_size决定。
            # 这里使用的插值方法是双线性插值（bilinear interpolation）。
            masks = F.interpolate(low_res_masks,
                                  (self.image_encoder.image_size, self.image_encoder.image_size),
                                  mode="bilinear", align_corners=False)
            input_size = image_record["image"].shape[-2:]
            original_size = image_record["original_size"]
            # 通过切片操作masks[..., : input_size[0], : input_size[1]]去除填充，
            # 其中input_size是模型输入的图像大小。
            masks = masks[..., :input_size[0], input_size[1]]
            masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
            # low_res_masks是模型输出的原始掩码，通过调用postprocess_masks函数进行后处理，得到经过调整和修正的masks。
            # input_size是模型输入的图像大小，通过image_record["image"].shape[-2:]获取。
            # original_size是原始图像的大小，从image_record["original_size"]中获取。
            # 经过postprocess_masks处理后的masks被与阈值self.mask_threshold进行比较，生成一个布尔掩码。
            # 其中大于阈值的像素为True，小于等于阈值的像素为False。
            masks_bool = masks > self.mask_shreshold
            outputs.append({
                "masks": masks_bool,
                "iou_predictions": iou_predictions,
                "low_res_logits": low_res_masks,
            })
        return outputs


def build_sam_vit_b(checkpoint=None,
                    encoder_emb_dim=768,
                    encoder_depth=12,
                    encoder_num_heads=12,
                    encoder_global_attn_idx=[2, 5, 8, 11]):
    prompt_emb_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_emb_size = image_size // vit_patch_size
    image_encoder = ImageEncoder(image_size=image_size,
                                 patch_size=vit_patch_size,
                                 embed_dim=encoder_emb_dim,
                                 depth=encoder_depth,
                                 num_heads=encoder_num_heads,
                                 mlp_ratio=4,
                                 qkv_bias=True,
                                 use_rel_pos=True,
                                 window_size=14,
                                 out_chans=prompt_emb_dim,
                                 global_attn_indexs=encoder_global_attn_idx,
                                 )
    prompt_encoder = PromptEncoder(embed_dim=prompt_emb_dim,
                                   image_embed_size=(image_emb_size, image_emb_size),
                                   input_img_size=(image_size, image_size),
                                   mask_in_chans=16)
    trm = TwoWayTransformer(depth=2, embed_dim=prompt_emb_dim, mlp_dim=2048, num_heads=8)
    mask_decoder = MaskDecoder(
        trm_dim=prompt_emb_dim,
        trm=trm,
        num_multitask_outputs=3,
        iou_head_dpt=3,
        iou_head_hidden_dim=256)
    pixel_mean = [123, 675, 116.28, 103.53]
    pixel_std = [58.395, 57.12, 57.375]
    sam = Sam(image_encoder, prompt_encoder, mask_decoder, pixel_mean, pixel_std)
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam


class ResizeLongestSide:
    # 用于调整图像和相关坐标的辅助类,将图像和坐标调整为指定的最长边的长度
    def __init__(self, target_length):
        self.target_length = target_length

    def apply_image(self, image: np.ndarray):
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]):
        # 将输入的坐标调整为指定最长边长度。需要提供原始图像大小。
        oldh, oldw = original_size
        newh, neww = self.get_preprocess_shape(original_size[0], original_size[1], self.target_length)

        coords = deepcopy(coords).astype(float)
        coords[...:0] = coords[..., 0] * (neww / oldw)
        coords[...:1] = coords[..., 1] * (newh / oldh)
        return coords

    def apply_box(self, boxes: np.ndarray, original_size: Tuple[int, ...]):
        # 应用框的调整
        # 将输入的框（boxes）调整为指定最长边长度。需要提供原始图像大小。
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor):
        # 应用Torch图像调整：将输入的Torch图像批次调整为指定最长边长度。
        # 使用了PyTorch中的插值方法。
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        result = F.interpolate(image, target_size, mode="bilinear", align_corners=False, antialias=True)
        return result

    def apply_coords_torch(self, coords: torch.Tensor, original: Tuple[int, ...]):
        oldh, oldw = original
        newh, neww = self.get_preprocess_shape(
            original[0], original[1], self.target_length
        )

        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (neww / oldw)
        coords[..., 1] = coords[..., 1] * (newh / oldh)
        return coords

    def apply_boxes_torch(self, boxes: torch.Tensor, original: Tuple[int, ...]):
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh, oldw, long_side_length):
        # 获取预处理形状：根据输入大小和目标最长边长度计算输出大小。
        scale = long_side_length * 1 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


class SamPredictor:
    def __init__(self, sam_model: Sam):
        super().__init__()
        self.model = sam_model
        self.trm = ResizeLongestSide(sam_model.image_encoder.image_size)
        self.reset_image()

    def set_image(self, image, image_format="RGB"):
        # 计算输入图像的image embedding
        assert image_format in ["RGB", "BGR"], f"image_format must be in ['RGB','BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        # 转换图片的type和shape
        input_image = self.trm.apply_image(image)
        input_image_torch = torch.as_torch(input_image,device=self.model.device)

    def reset_image(self):
        self.is_image_set = False
        self.features = None
        self.originalh = None
        self.originalw = None
        self.inputh = None
        self.inputw = None


if __name__ == '__main__':
    sam_checkpoint = None
    model_type = "vit_b"
    sam = build_sam_vit_b(checkpoint=sam_checkpoint)
    print(sam)

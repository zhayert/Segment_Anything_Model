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
from einops import rearrange

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
        self.register_buffer("pixel_mean", rearrange(torch.Tensor(pixel_mean), 'a -> a 1 1'), False)
        self.register_buffer("pixel_std", rearrange(torch.Tensor(pixel_std), 'a -> a 1 1'), False)

    @property
    def device(self):
        print(self.pixel_mean.device)
        return self.pixel_mean.device

    @torch.no_grad()
    def forward(self, batch_input, multimask_output):
        # You can try it
        # for i in batch_input:
        #     x = (i - self.pixel_mean) / self.pixel_std
        #     h, w = x.shape[-2:]
        #     padh = self.image_encoder.image_size - h
        #     padw = self.image_encoder.image_size - w
        #     x = F.pad(x, (0, padw, 0, padh))
        #     input_images = torch.stack(x, dim=0)

        input_images = torch.stack([self.preprocess(x["image"]) for x in batch_input], dim=0)
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

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.image_size - h
        padw = self.image_encoder.image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.image_encoder.image_size, self.image_encoder.image_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks


def build_sam_vit_b(
        checkpoint=None,  # 预训练权重
        encoder_emb_dim=768,  # 图像编码Channel
        encoder_depth=12,  # 主体编码器的个数
        encoder_num_heads=12,  # Attention中head的个数
        encoder_global_attn_idx=[2, 5, 8, 11]  # 需要将pos_embed添加到Encoder Block
):
    prompt_emb_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_emb_size = image_size // vit_patch_size
    image_encoder = ImageEncoder(
        image_size=image_size,  # 输入图像的尺寸
        patch_size=vit_patch_size,  # patch的大小
        embed_dim=encoder_emb_dim,  # 图像编码的channel
        depth=encoder_depth,  # 主体编码器的个数
        num_heads=encoder_num_heads,  # attention中head的个数
        mlp_ratio=4,  # mlp中channel的缩放比
        qkv_bias=True,  # qkv全连阶层的偏置
        use_rel_pos=True,  # 是否要将rel_pos_embed加入到embed中
        window_size=14,  # attention中窗口的大小
        out_chans=prompt_emb_dim,  # 输出的channel
        global_attn_indexs=encoder_global_attn_idx,  # 需要将 rel_pos_embed
    )
    prompt_encoder = PromptEncoder(
        embed_dim=prompt_emb_dim,  # 提示编码的channel(和image_encoder输出的channel保持一致，后续融合)
        image_embed_size=(image_emb_size, image_emb_size),  # mask的编码尺寸(和image_encoder输出channel一致)
        image_size=(image_size, image_size),  # 输入图像的标准尺寸
        mask_in_chans=16  # 对输入mask编码的通道数
    )
    # MaskDeco由多个重复堆叠TwoWayAttention Block和1个Multi-Head Attention组成。
    trm = TwoWayTransformer(depth=2, embed_dim=prompt_emb_dim, mlp_dim=2048, num_heads=8)
    mask_decoder = MaskDecoder(
        trm_dim=prompt_emb_dim,  # transformer的channnel
        trm=trm,  # 用于预测mask网络的transformer
        num_multitask_outputs=3,  # 消除掩码歧义预测的掩码数
        iou_head_dpt=3,  # MLP的深度，MLP用于预测掩码质量
        iou_head_hidden_dim=256  # MLP隐藏的channel
    )
    # 图像预处理的参数，进行归一化处理，将像素值落在一个较小的范围内，有助于提高模型的训练效果。
    pixel_mean = [123.675, 116.28, 103.53]
    pixel_std = [58.395, 57.12, 57.375]
    sam = Sam(image_encoder, prompt_encoder, mask_decoder, pixel_mean, pixel_std)
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam


class SamPredictor:
    def __init__(self, sam_model: Sam):
        super().__init__()
        self.model = sam_model  # sam mask预测模型
        self.trm = ResizeLongestSide(sam_model.image_encoder.image_size)  # 数据预处理
        self.reset_image()  # 图像相关的数据信息

    # self.features保存图片经过Image Encoder后的特征数据
    # self.is_image_set是一个信号信息，用来表示self.features是否已经保存了特征数据。
    # 在刚初始化时，self.features是none，self.is_image_set便是false。
    def reset_image(self):
        # 图像设置flag
        self.is_image_set = False
        # 图像编码特征
        self.features = None
        self.originalh = None
        self.originalw = None
        self.inputh = None
        self.inputw = None

    # 首先确认输入是否是RGB或BGR三通道图像，将BGR图像统一为RGB，
    # 对图像尺寸(apply_image)和channel顺序作出调整满足神经网络的输入要求。
    # 计算输入图像的image embedding
    def set_image(self, image, image_format="RGB"):
        # 图像不是['RGB', 'BGR']格式则报错
        assert image_format in ["RGB", "BGR"], f"image_format must be in ['RGB','BGR'], is {image_format}."
        # 检查输入图像format是否与模型的期望format相同
        # H, W, C
        if image_format != self.model.image_format:
            image = image[..., ::-1]  # H,W,C中 C通道的逆序RGB-->BGR

        # 转换图片的shape转换到期望输入模型的形状
        input_image = self.trm.apply_image(image)
        # torch浅拷贝，转tensor
        input_image_torch = torch.as_tensor(input_image, device=self.model.device)
        # permute H,W,C-->C,H,W
        # contiguous 连续内存
        # [None,:,:,:] C, H, W --> 1, C, H, W
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        # 调用另一个方法以设置torch图像
        self.set_torch_image(input_image_torch, image.shape[:2])

    # 用padding填补缩放后的图片，在H和W满足神经网络需要的标准尺寸，
    # 而后通过image_encoder模型获得图像特征数据并保存在self.features中，
    # 同时self.is_image_set设为true
    @torch.no_grad()
    def set_torch_image(self, new_image: torch.Tensor, old_image_size: Tuple[int, ...]):
        # 满足输入是四个维度且为B,C,H,W
        assert (len(new_image.shape) == 4
                and new_image.shape[1] == 3
                and max(*new_image.shape[2:]) == self.model.image_encoder.image_size
                ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.image_size}"
        self.reset_image()

        # 存储原始图像尺寸和torch图像的尺寸
        self.old_size = old_image_size
        self.input_size = tuple(new_image.shape[-2:])

        # torch图像进行padding
        input_image = self.model.preprocess(new_image)
        # image_encoder网络模块对图像进行编码
        self.features = self.model.image_encoder(input_image)

        # 标记图像已经被设置
        self.is_image_set = True

    # 用于模型的预测，根据给定的point_coords、point_labels、box、mask_input。
    # 对输入到模型中进行预测的数据(标记点apply_coords和标记框apply_boxes)进行一个预处理，
    # 并接受和处理模型返回的预测结果。
    def pred(
            self,
            point_coords=None,  # 标记点坐标
            point_labels=None,  # 标记点标签
            box=None,  # 标记框坐标
            mask_input=None,  # 输入mask
            multimask_output=True,  # 输出多个mask供选择
            return_logits=False  # True返回logits，False返回阈值处理的二进制编码
    ):
        # 没有设置图像会报错
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # prompt input转换为torch
        # 根据需要对输入参数进行坐标和尺寸的变换，确保输入符合模型的期望格式。
        coords_torch, labels_torch, box_torch, mask_torch = None, None, None, None
        if point_coords is not None:
            # 标记点坐标对应的标记点标签不能为空
            assert (point_labels is not None), "point_labels must be supplied if point_coords is supplied"
            point_coords = self.trm.apply_coords(point_coords, self.old_size)
            # 图像改变了原始尺寸,所以对应的点位置也会发生改变
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.model.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.float, device=self.model.device)
            # 增加维度
            # coords_torch: N, 2 --> 1, N, 2
            # labels_torch: N --> 1, N
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.trm.apply_box(box, self.old_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.model.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            # mask np-->tensor
            mask_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.model.device)
            # 增加维度 1,H,W-->B,1,H,W
            mask_torch = mask_torch[None, :, :, :]

        # 输入数据预处理完成，可以输入到网络中。
        masks, iou_pred, low_res_masks = self.pred_torch(
            coords_torch, labels_torch, box_torch, mask_torch, multimask_output, return_logits
        )

        # mask
        # 将预测结果从PyTorch张量转换为NumPy数组，并返回三个主要的输出：
        # 输出masks：形状为CxHxW，其中C是掩膜的数量，H和W是原始图像的高度和宽度。
        masks_np = masks[0].detach().cpu().numpy()

        # score
        # 模型对每个掩膜的预测质量：一个长度为C的数组。
        iou_pred_np = iou_pred[0].detach().cpu().numpy()
        # 低分辨率掩膜：形状为CxHxW，其中C是掩膜的数量，H和W通常为256。这些低分辨率的logits可以作为下一次迭代的掩膜输入。
        low_res_masks_np = low_res_masks[0].detach().cpu().numpy()

        return masks_np, iou_pred_np, low_res_masks_np

    # 输入数据经过预处理后输入到模型中预测结果。
    @torch.no_grad()
    def pred_torch(self,
                   point_coords,
                   point_labels,
                   boxes=None,
                   mask_input=None,
                   multimask_output=True,
                   return_logits=False):
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction")

        # 将输入提示（points、boxes、mask_input）嵌入模型
        # 得到稀疏（sparse_embeddings）和密集（dense_embeddings）的嵌入。
        if point_coords is not None:
            # 绑定标记点和标记点标签
            points = (point_coords, point_labels)
        else:
            points = None

        # --- Prompt Encoder ---
        sparse_embed, dense_embed = self.model.prompt_encoder(
            points=points, boxes=boxes, masks=mask_input
        )
        # --- Prompt Encoder ---

        # --- Mask Decoder ---
        # 使用模型的掩膜解码器（mask_decoder）来预测masks和iou_pred。
        low_res_mask, iou_pred = self.model.mask_decoder(
            image_embed=self.features,
            image_pos_embed=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embed=sparse_embed,
            dense_prompt_embed=dense_embed,
            multimask_output=multimask_output
        )
        # --- Mask Decoder ---

        # 上采样mask掩膜到原始图片尺寸
        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_mask, self.input_size, self.old_size)

        if not return_logits:
            masks = masks > self.model.mask_shreshold

        return masks, iou_pred, low_res_mask


# ResizeLongestSide是专门用来处理图片、标记点和标记框的工具类。
class ResizeLongestSide:
    # 设置了所有输入到神经网络的标准图片尺寸
    def __init__(self, target_length):
        self.target_length = target_length

    # 原图尺寸根据标准尺寸计算调整(get_preprocess_shape)得新尺寸。
    def apply_image(self, image: np.ndarray):
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        # to_pil_image将numpy装变为PIL.Image,而后resize
        # 不直接使用resize的目的是为了不破坏原图片中各个物体的比例关系。
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, old_size: Tuple[int, ...]):
        oldh, oldw = old_size
        # 图像改变了原始尺寸,所以对应的标记点坐标位置也会发生改变
        newh, neww = self.get_preprocess_shape(
            old_size[0], old_size[1], self.target_length)
        # 深拷贝coords
        coords = deepcopy(coords).astype(float)
        # 改变对应标记点的坐标
        coords[..., 0] = coords[..., 0] * (neww / oldw)
        coords[..., 1] = coords[..., 1] * (newh / oldh)
        return coords

    # 图像改变了原始尺寸，对应的标记框坐标位置也要改变(get_preprocess_shape)。
    def apply_box(self, boxes: np.ndarray, original_size: Tuple[int, ...]):
        # 图像改变了原始尺寸,所以对应的框坐标位置也会发生改变
        # reshape: N,4-->N,2,2
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        # reshape: N,2,2-->N,4
        return boxes.reshape(-1, 4)

    # def apply_image_torch(self, image: torch.Tensor):
    #     # 应用Torch图像调整：将输入的Torch图像批次调整为指定最长边长度。
    #     # 使用了PyTorch中的插值方法。
    #     target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
    #     result = F.interpolate(image, target_size, mode="bilinear", align_corners=False, antialias=True)
    #     return result
    #
    # def apply_coords_torch(self, coords: torch.Tensor, original: Tuple[int, ...]):
    #     oldh, oldw = original
    #     newh, neww = self.get_preprocess_shape(
    #         original[0], original[1], self.target_length
    #     )
    #
    #     coords = deepcopy(coords).to(torch.float)
    #     coords[..., 0] = coords[..., 0] * (neww / oldw)
    #     coords[..., 1] = coords[..., 1] * (newh / oldh)
    #     return coords
    #
    # def apply_boxes_torch(self, boxes: torch.Tensor, original: Tuple[int, ...]):
    #     boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original)
    #     return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh, oldw, long_side_length):
        # 获取预处理形状：根据输入大小和目标最长边长度计算输出大小。
        scale = long_side_length * 1 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        # 四舍五入？
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


if __name__ == '__main__':
    sam_checkpoint = None
    model_type = "vit_b"
    sam = build_sam_vit_b(checkpoint=sam_checkpoint)
    print(sam)

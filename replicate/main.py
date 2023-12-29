import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from sam import build_sam_vit_b, SamPredictor


# from predictor_sam import SamPredictor
# from build_sam import sam_model_registry
torch.manual_seed(42)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


image = cv2.imread('../images_input/msg-1001890926312-10348.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image)
print(image.shape)  # (880, 1195, 3)
# plt.imshow(image)
# plt.show()


sam_checkpoint = r"..\..\Segment_Anything_Model\pretrained\sam_vit_b_01ec64.pth"
# model_type = "vit_b"

device = "cuda"

sam = build_sam_vit_b(checkpoint=sam_checkpoint)
# sam = build_sam_vit_b(checkpoint=sam_checkpoint)
sam.to(device=device)

pred = SamPredictor(sam)
pred.set_image(image, "RGB")

# One point
# input_point = np.array([[317, 352]])
# input_label = np.array([1])

# Two points
input_point = np.array([[320, 600], [800, 600]])
input_label = np.array([1, 1])

# A point and a background point
# input_point = np.array([[800, 400], [320, 600]])
# input_label = np.array([1, 0])

# One Box
# input_box = np.array([425, 600, 700, 875])

masks, scores, logits = pred.pred(
    point_coords=input_point,
    point_labels=input_label,
    # box=input_box[None,:],
    multimask_output=True
)

print(masks.shape)
print(scores)
print(logits.shape)

fig, axs = plt.subplots(1, 3, figsize=(12, 6))

for i, (mask, score) in enumerate(zip(masks, scores)):
    axs[i].imshow(image)
    show_mask(mask, axs[i])
    show_points(input_point, input_label, axs[i])
    axs[i].set_title("Mask{}, Score: {:.3f}".format(i + 1, score), fontsize=18)
    plt.axis("on")

plt.show()

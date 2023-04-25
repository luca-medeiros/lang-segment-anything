import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks

MIN_AREA = 100


def load_image(image_path: str):
    return Image.open(image_path).convert("RGB")


def draw_image(image, masks, boxes, labels, alpha=0.4):
    image = torch.from_numpy(image).permute(2, 0, 1)
    if len(boxes) > 0:
        image = draw_bounding_boxes(image, boxes, colors=['red'] * len(boxes), labels=labels, width=2)
    if len(masks) > 0:
        image = draw_segmentation_masks(image, masks=masks, colors=['cyan'] * len(masks), alpha=alpha)
    return image.numpy().transpose(1, 2, 0)


def get_contours(mask):
    if len(mask.shape) > 2:
        mask = np.squeeze(mask, 0)
    mask = mask.astype(np.uint8)
    mask *= 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    effContours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > MIN_AREA:
            effContours.append(c)
    return effContours


def contour_to_points(contour):
    pointsNum = len(contour)
    contour = contour.reshape(pointsNum, -1).astype(np.float32)
    points = [point.tolist() for point in contour]
    return points


def generate_labelme_json(binary_masks, labels, image_size, image_path=None):
    """Generate a LabelMe format JSON file from binary mask tensor.

    Args:
        binary_masks: Binary mask tensor of shape [N, H, W].
        labels: List of labels for each mask.
        image_size: Tuple of (height, width) for the image size.
        image_path: Path to the image file (optional).

    Returns:
        A dictionary representing the LabelMe JSON file.
    """
    num_masks = binary_masks.shape[0]
    binary_masks = binary_masks.numpy()

    json_dict = {
        "version": "4.5.6",
        "imageHeight": image_size[0],
        "imageWidth": image_size[1],
        "imagePath": image_path,
        "flags": {},
        "shapes": [],
        "imageData": None
    }

    # Loop through the masks and add them to the JSON dictionary
    for i in range(num_masks):
        mask = binary_masks[i]
        label = labels[i]
        effContours = get_contours(mask)

        for effContour in effContours:
            points = contour_to_points(effContour)
            shape_dict = {
                "label": label,
                "line_color": None,
                "fill_color": None,
                "points": points,
                "shape_type": "polygon"
            }

            json_dict["shapes"].append(shape_dict)

    return json_dict

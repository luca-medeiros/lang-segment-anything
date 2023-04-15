import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from s3fs import S3FileSystem
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks

MIN_AREA = 100
IMG_EXT = [".jpg", ".jpeg", ".png", ".bmp"]


def results_to_df(results):
    data = []
    for result in results:
        image_path = result['image']['file_path']
        img_filename = os.path.basename(image_path)
        width = result['image']['width']
        height = result['image']['height']

        for _, box, label, score in result['outputs']:
            bbox_x, bbox_y, bbox_w, bbox_h = box
            data.append({
                'img_filename': img_filename,
                'bbox_x': bbox_x,
                'bbox_y': bbox_y,
                'bbox_w': bbox_w,
                'bbox_h': bbox_h,
                'label': label,
                'score': score,
            })

    df = pd.DataFrame(data)
    return df


def load_image(image_path):

    if image_path.startswith("s3"):
        fs = S3FileSystem()
        with fs.open(image_path, "rb") as f:
            image = Image.open(f).convert("RGB")
    else:
        with open(image_path, "rb") as f:
            image = Image.open(f).convert("RGB")

    return image


def draw_image(image, masks, boxes, labels, alpha=0.4):
    image = torch.from_numpy(image).permute(2, 0, 1)
    image = draw_bounding_boxes(image, boxes, colors=['red'] * len(boxes), labels=labels, width=2)
    image = draw_segmentation_masks(image, masks=masks, colors=['cyan'] * len(boxes), alpha=alpha)
    return image.numpy().transpose(1, 2, 0)


def get_image_paths(data_folder: str):
    image_paths = []

    # Check if it's a local directory or an S3 URI
    if data_folder.startswith("s3"):
        path_parts = data_folder.replace("s3://", "").split("/")
        bucket_name = path_parts.pop(0)
        prefix = "/".join(path_parts)

        fs = S3FileSystem()

        s3_image_uris = fs.glob(f"{bucket_name}/{prefix}/*")
        for uri in s3_image_uris:
            _, ext = os.path.splitext(uri)
            if ext.lower() in IMG_EXT:
                image_paths.append(f"s3://{uri}")
    else:
        # It's a local directory
        data_folder = Path(data_folder)
        for ext in IMG_EXT:
            image_paths.extend([str(p.resolve()) for p in data_folder.glob('*' + ext)])

    return image_paths


def mask_to_polygon(mask):
    """Convert a binary mask to COCO polygon format."""
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    # Find contours in the binary mask
    contours = get_contours(mask)

    segmentation = []
    for contour in contours:
        poly = contour.flatten().tolist()
        if len(poly) > 4:
            segmentation.append(poly)

    return segmentation


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

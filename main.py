import os
import torch
import cv2
import time
import numpy as np
import gradio as gr

from PIL import Image
from random import randint
from typing import List

from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import load_image, predict

from segment_anything import build_sam, SamPredictor

from huggingface_hub import hf_hub_download


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model


def draw_masks(image: np.ndarray, masks: List[np.ndarray], alpha: float = 0.4) -> np.ndarray:
    for mask in masks:
        color = [randint(130, 255) for _ in range(3)]

        # draw mask overlay
        colored_mask = np.expand_dims(mask[0], 0).repeat(3, axis=0)
        colored_mask = np.moveaxis(colored_mask, 0, -1)
        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
        image_overlay = masked.filled()
        image = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

        # draw contour
        contours, _ = cv2.findContours(np.uint8(mask[0]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (70, 0, 0), 2)
    return image


def run(box_threshold, text_threshold, image_path, text_prompt):
    st = time.time()
    image_source, image = load_image(image_path)
    boxes = run_dino(image, text_prompt, box_threshold, text_threshold)
    print('dino', time.time() - st)
    print(boxes)
    if len(boxes) == 0:
        return Image.fromarray(np.uint8(image_source)).convert("RGB")
    masks = run_sam(image_source, boxes)
    print('sam', time.time() - st)
    image = draw_masks(image_source, masks)
    print('draw', time.time() - st)
    image = Image.fromarray(np.uint8(image)).convert("RGB")
    return image


def run_dino(image_trans, text_prompt, box_threshold, text_threshold):
    boxes, logits, phrases = predict(model=groundingdino_model,
                                     image=image_trans,
                                     caption=text_prompt,
                                     box_threshold=box_threshold,
                                     text_threshold=text_threshold)
    return boxes


def run_sam(image_source, boxes):
    sam_predictor.set_image(image_source)
    # box: normalized box xywh -> unnormalized xyxy
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2])
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(sam_predictor.device),
        multimask_output=False,
    )
    return masks.cpu()


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Slider(0, 1, value=0.3, label="box_threshold"),
        gr.Slider(0, 1, value=0.25, label="text_threshold"),
        gr.Image(type="filepath"),
        "text",
    ],
    outputs="image",
    allow_flagging="never",
    title="Segment Anything with Grounding DINO",
    examples=[
        [
            0.36,
            0.25,
            os.path.join(os.path.dirname(__file__), "assets", "fruits.jpg"),
            "kiwi",
        ],
        [
            0.3,
            0.25,
            os.path.join(os.path.dirname(__file__), "assets", "car.jpeg"),
            "wheel",
        ],
        [
            0.3,
            0.25,
            os.path.join(os.path.dirname(__file__), "assets", "food.jpg"),
            "food",
        ],
    ],
)

ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)
sam_checkpoint = 'sam_vit_h_4b8939.pth'
sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to('cuda'))

if __name__ == "__main__":
    demo.launch(share=True)
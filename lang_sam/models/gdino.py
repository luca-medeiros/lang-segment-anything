import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


class GDINO:
    def __init__(self):
        self.build_model()

    def build_model(self, ckpt_path: str | None = None):
        model_id = "IDEA-Research/grounding-dino-base"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)

    def predict(
        self,
        pil_image: Image.Image,
        text_prompt: str,
        box_threshold: float,
        text_threshold: float,
    ):
        if text_prompt[-1] != ".":
            text_prompt += "."
        inputs = self.processor(images=pil_image, text=text_prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[pil_image.size[::-1]],
        )
        return results[0]

    def predict_batch(
        self,
        pil_images: list[Image.Image],
        text_prompt: str,
        box_threshold: float,
        text_threshold: float,
    ):
        raise NotImplementedError("GDINO does not support batch prediction yet")

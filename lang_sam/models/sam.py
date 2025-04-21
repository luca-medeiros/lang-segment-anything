import numpy as np
import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor

from lang_sam.models.utils import DEVICE

SAM_MODELS = {
    "sam2.1_hiera_tiny": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
    },
    "sam2.1_hiera_small": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
    },
    "sam2.1_hiera_base_plus": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    },
    "sam2.1_hiera_large": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
    },
}


class SAM:
    def build_model(self, sam_type: str, ckpt_path: str | None = None, device=DEVICE):
        self.sam_type = sam_type
        self.ckpt_path = ckpt_path
        cfg = compose(config_name=SAM_MODELS[self.sam_type]["config"], overrides=[])
        OmegaConf.resolve(cfg)
        self.model = instantiate(cfg.model, _recursive_=True)
        self._load_checkpoint(self.model)
        self.model = self.model.to(device)
        self.model.eval()
        self.mask_generator = SAM2AutomaticMaskGenerator(self.model)
        self.predictor = SAM2ImagePredictor(self.model)

    def _load_checkpoint(self, model: torch.nn.Module):
        if self.ckpt_path is None:
            checkpoint_url = SAM_MODELS[self.sam_type]["url"]
            state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["model"]
        else:
            checkpoint_url = self.ckpt_path  # Ensure checkpoint_url is defined
            state_dict = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)["model"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            raise ValueError(
                f"Problem loading SAM please make sure you have the right model type: {self.sam_type} \
                and a working checkpoint: {checkpoint_url}. Recommend deleting the checkpoint and \
                re-downloading it. Error: {e}"
            )

    def generate(self, image_rgb: np.ndarray) -> list[dict]:
        """
        Output format
        SAM2AutomaticMaskGenerator returns a list of masks, where each mask is a dict containing various information
        about the mask:

        segmentation - [np.ndarray] - the mask with (W, H) shape, and bool type
        area - [int] - the area of the mask in pixels
        bbox - [List[int]] - the boundary box of the mask in xywh format
        predicted_iou - [float] - the model's own prediction for the quality of the mask
        point_coords - [List[List[float]]] - the sampled input point that generated this mask
        stability_score - [float] - an additional measure of mask quality
        crop_box - List[int] - the crop of the image used to generate this mask in xywh format
        """

        sam2_result = self.mask_generator.generate(image_rgb)
        return sam2_result

    def predict(self, image_rgb: np.ndarray, xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.predictor.set_image(image_rgb)
        masks, scores, logits = self.predictor.predict(box=xyxy, multimask_output=False)
        if len(masks.shape) > 3:
            masks = np.squeeze(masks, axis=1)
        return masks, scores, logits

    def predict_batch(
        self,
        images_rgb: list[np.ndarray],
        xyxy: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        self.predictor.set_image_batch(images_rgb)

        masks, scores, logits = self.predictor.predict_batch(box_batch=xyxy, multimask_output=False)

        masks = [np.squeeze(mask, axis=1) if len(mask.shape) > 3 else mask for mask in masks]
        scores = [np.squeeze(score) for score in scores]
        logits = [np.squeeze(logit, axis=1) if len(logit.shape) > 3 else logit for logit in logits]
        return masks, scores, logits

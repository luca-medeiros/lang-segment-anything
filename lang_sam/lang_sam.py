import numpy as np

from lang_sam.models.gdino import GDINO
from lang_sam.models.sam import SAM, SAM_MODELS


class LangSAM:
    def __init__(self, sam_type="sam2_hiera_small", ckpt_path: str | None = None):
        self.sam_type = sam_type
        self.sam = SAM()
        self.sam.build_model(sam_type, ckpt_path)
        self.gdino = GDINO()
        self.gdino.build_model()

    def predict(self, image_pil, text_prompt, box_threshold=0.3, text_threshold=0.25):
        results: dict = self.gdino.predict(image_pil, text_prompt, box_threshold, text_threshold)
        results["masks"] = []
        results["mask_scores"] = []
        results["mask_logits"] = []
        if len(results["labels"]):
            results["boxes"] = results["boxes"].cpu().numpy()
            results["scores"] = results["scores"].cpu().numpy()

            image_array = np.asarray(image_pil)
            masks, mask_scores, logits = self.sam.predict(image_array, xyxy=results["boxes"])
            results["masks"] = masks
            results["mask_scores"] = mask_scores
            results["mask_logits"] = logits
        return results

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from lang_sam.models.utils import DEVICE


class GDINO:
    def build_model(self, model_ckpt_path: str | None = None, processor_ckpt_path: str | None = None, device=DEVICE):
        if not model_ckpt_path or not processor_ckpt_path: # indicates that we somehow able to load the model from internet
            model_id = "IDEA-Research/grounding-dino-base"
            print(f"One or both local paths not provided. Loading from Hugging Face Hub: {model_id}")
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

        else:
            print(f"Attempting to load processor from local path: {processor_ckpt_path}")
            self.processor = AutoProcessor.from_pretrained(
                processor_ckpt_path,
                local_files_only=True,        # never goes online
                trust_remote_code=True,       # Grounding-DINO uses custom code
            )
            print(f"Attempting to load model from local path: {model_ckpt_path}")
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_ckpt_path,
                local_files_only=True,
                trust_remote_code=True,
                use_safetensors=True,
            ).to(device)

    def predict(
        self,
        images_pil: list[Image.Image],
        texts_prompt: list[str],
        box_threshold: float,
        text_threshold: float,
    ) -> list[dict]:
        texts_prompt = [prompt if prompt[-1] == "." else prompt + "." for prompt in texts_prompt]
        inputs = self.processor(
            images=images_pil, text=texts_prompt, padding=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold,
            text_threshold=text_threshold,
            target_sizes=[k.size[::-1] for k in images_pil],
        )
        return results


if __name__ == "__main__":
    gdino = GDINO()
    gdino.build_model()
    out = gdino.predict(
        [Image.open("./assets/car.jpeg"), Image.open("./assets/car.jpeg")],
        ["wheel", "wheel"],
        0.3,
        0.25,
    )
    print(out)

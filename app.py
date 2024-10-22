import os
from io import BytesIO

import gradio as gr
import requests
from PIL import Image

from lang_sam import SAM_MODELS
from lang_sam.server import PORT, server


def inference(sam_type, box_threshold, text_threshold, image, text_prompt):
    """Gradio function that makes a request to the /predict LitServe endpoint."""
    url = f"http://localhost:{PORT}/predict"  # Adjust port if needed

    # Prepare the multipart form data
    with open(image, "rb") as img_file:
        files = {
            "image": img_file,
        }
        data = {
            "sam_type": sam_type,
            "box_threshold": str(box_threshold),
            "text_threshold": str(text_threshold),
            "text_prompt": text_prompt,
        }

        try:
            response = requests.post(url, files=files, data=data)
        except Exception as e:
            print(f"Request failed: {e}")
            return None

    if response.status_code == 200:
        try:
            output_image = Image.open(BytesIO(response.content)).convert("RGB")
            return output_image
        except Exception as e:
            print(f"Failed to process response image: {e}")
            return None
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")
        return None


with gr.Blocks(title="lang-sam") as blocks:
    with gr.Row():
        sam_model_choices = gr.Dropdown(choices=list(SAM_MODELS.keys()), label="SAM Model", value="sam2.1_hiera_small")
        box_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, label="Box Threshold")
        text_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.25, label="Text Threshold")
    with gr.Row():
        image_input = gr.Image(type="filepath", label="Input Image")
        output_image = gr.Image(type="pil", label="Output Image")
    text_prompt = gr.Textbox(lines=1, label="Text Prompt")

    submit_btn = gr.Button("Run Prediction")

    submit_btn.click(
        fn=inference,
        inputs=[sam_model_choices, box_threshold, text_threshold, image_input, text_prompt],
        outputs=output_image,
    )

    examples = [
        [
            "sam2.1_hiera_small",
            0.32,
            0.25,
            os.path.join(os.path.dirname(__file__), "assets", "fruits.jpg"),
            "kiwi. watermelon. blueberry.",
        ],
        [
            "sam2.1_hiera_small",
            0.3,
            0.25,
            os.path.join(os.path.dirname(__file__), "assets", "car.jpeg"),
            "wheel.",
        ],
        [
            "sam2.1_hiera_small",
            0.3,
            0.25,
            os.path.join(os.path.dirname(__file__), "assets", "food.jpg"),
            "food.",
        ],
    ]

    gr.Examples(
        examples=examples,
        inputs=[sam_model_choices, box_threshold, text_threshold, image_input, text_prompt],
        outputs=output_image,
    )

server.app = gr.mount_gradio_app(server.app, blocks, path="/gradio")

if __name__ == "__main__":
    print(f"Starting LitServe and Gradio server on port {PORT}...")
    server.run(port=PORT)

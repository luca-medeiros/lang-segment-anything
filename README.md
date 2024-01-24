# Language Segment-Anything

Language Segment-Anything is an open-source project that combines the power of instance segmentation and text prompts to generate masks for specific objects in images. Built on the recently released Meta model, segment-anything, and the GroundingDINO detection model, it's an easy-to-use and effective tool for object detection and image segmentation.

![person.png](/assets/outputs/person.png)

## Features

- Zero-shot text-to-bbox approach for object detection.
- GroundingDINO detection model integration.
- Easy deployment using the Lightning AI app platform.
- Customizable text prompts for precise object segmentation.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- torch (tested 2.0)
- torchvision

### Installation

```
pip install torch torchvision
pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git
```

Or
Clone the repository and install the required packages:

```
git clone https://github.com/luca-medeiros/lang-segment-anything && cd lang-segment-anything
pip install torch torchvision
pip install -e .
```
Or use Conda
Create a Conda environment from the `environment.yml` file:
```
conda env create -f environment.yml
# Activate the new environment:
conda activate lsa
```

#### Docker Installation

Build and run the image.

	```
	docker build --tag lang-segment-anything:latest .
	docker run --gpus all -it lang-segment-anything:latest
	```

If you want a shared folder you can add a volume with `-v <host_source_dir>:<container_dest_dir>` example: `-v ./data:/workspace/data`


### Usage

To run the Lightning AI APP:

`lightning run app app.py`

Use as a library:

```python
from PIL import Image
from lang_sam import LangSAM

model = LangSAM()
image_pil = Image.open("./assets/car.jpeg").convert("RGB")
text_prompt = "wheel"
masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
```

Use with custom checkpoint:

First download a model checkpoint. 

```python
from PIL import Image
from lang_sam import LangSAM

model = LangSAM("<model_type>", "<path/to/checkpoint>")
image_pil = Image.open("./assets/car.jpeg").convert("RGB")
text_prompt = "wheel"
masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
```

## Examples

![car.png](/assets/outputs/car.png)

![kiwi.png](/assets/outputs/kiwi.png)

![person.png](/assets/outputs/person.png)

## Roadmap

Future goals for this project include:

1. **FastAPI integration**: To streamline deployment even further, we plan to add FastAPI code to our project, making it easier for users to deploy and interact with the model.

1. **Labeling pipeline**: We want to create a labeling pipeline that allows users to input both the text prompt and the image and receive labeled instance segmentation outputs. This would help users efficiently generate results for further analysis and training.

1. **Implement CLIP version**: To (maybe) enhance the model's capabilities and performance, we will explore the integration of OpenAI's CLIP model. This could provide improved language understanding and potentially yield better instance segmentation results.

## Acknowledgments

This project is based on the following repositories:

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [Segment-Anything](https://github.com/facebookresearch/segment-anything)
- [Lightning AI](https://github.com/Lightning-AI/lightning)

## License

This project is licensed under the Apache 2.0 License

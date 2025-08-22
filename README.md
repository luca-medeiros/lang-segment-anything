# Language Segment-Anything

Language Segment-Anything is an open-source project that combines the power of instance segmentation and text prompts to generate masks for specific objects in images. Built on the recently released Meta model, Segment Anything Model 2, and the GroundingDINO detection model, it's an easy-to-use and effective tool for object detection and image segmentation.

![person.png](/assets/outputs/person.png)

## Features

- Zero-shot text-to-bbox approach for object detection.
- GroundingDINO detection model integration.
- SAM 2.1
- Batch inference support.
- Easy endpoint deployment using the Lightning AI litserve platform.
- Customizable text prompts for precise object segmentation.

## Getting Started

### Prerequisites

- Python 3.10 or higher

### Installation

#### Installing PyTorch Dependencies

Before installing `lang-sam`, please install PyTorch using the following command:

```bash

pip install torch==2.4.1 torchvision==0.19.1 --extra-index-url https://download.pytorch.org/whl/cu124

```

```bash

pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git

```

Or
Clone the repository and install the required packages:

```bash

git clone https://github.com/luca-medeiros/lang-segment-anything && cd lang-segment-anything
pip install -e .

```

#### Docker Installation

Build and run the image.

```bash

git clone https://github.com/luca-medeiros/lang-segment-anything && cd lang-segment-anything
docker build --tag lang-segment-anything:latest .
docker run --gpus all -p 8000:8000 lang-segment-anything:latest

```

### Usage

To run the gradio APP:

`python app.py`
And open `http://0.0.0.0:8000/gradio`

Use as a library:

```python
from PIL import Image
from lang_sam import LangSAM

model = LangSAM()
image_pil = Image.open("./assets/car.jpeg").convert("RGB")
text_prompt = "wheel."
results = model.predict([image_pil], [text_prompt])
```

If desired, arguments below can be passed for offline operations:
```python
LangSAM(sam_ckpt_path,              # path for segment anything model
        gdino_model_ckpt_path,      # path for grounding dino's model checkpoint
        gdino_processor_ckpt_path   # path for grounding dino's processor checkpoint
)
```

## Examples

![car.png](/assets/outputs/car.png)

![fruits.png](/assets/outputs/fruits.png)

## Acknowledgments

This project is based on/used the following repositories:

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [Segment-Anything](https://github.com/facebookresearch/segment-anything-2)
- [LitServe](https://github.com/Lightning-AI/LitServe/)
- [Supervision](https://github.com/roboflow/supervision)

## License

This project is licensed under the Apache 2.0 License

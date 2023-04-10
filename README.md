# Language Segment-Anything
Image + Text prompt -> GroundingDINO -> BBox -> Sam

![car.png](/assets/outputs/car.png)
![kiwi.png](/assets/outputs/kiwi.png)
![food.png](/assets/outputs/food.png)

## Running

To run the gradio demo:

    git clone https://github.com/luca-medeiros/lang-segment-anything
    cd lang-segment-anything
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    pip install -r requirements.txt
    python main.py

### Based on

https://github.com/IDEA-Research/GroundingDINO

https://github.com/facebookresearch/segment-anything

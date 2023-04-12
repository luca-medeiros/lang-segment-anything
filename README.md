# Language Segment-Anything
Image + Text prompt -> GroundingDINO -> BBox -> SAM -> masks

![car.png](/assets/outputs/car.png)
![kiwi.png](/assets/outputs/kiwi.png)
![food.png](/assets/outputs/food.png)

## Running

To run the Lightning AI APP:

    git clone https://github.com/luca-medeiros/lang-segment-anything
    cd lang-segment-anything
    pip install -r requirements.txt
    lightning run app app.py

### Based on

https://github.com/IDEA-Research/GroundingDINO

https://github.com/facebookresearch/segment-anything

https://github.com/Lightning-AI/lightning

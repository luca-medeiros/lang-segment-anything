## Runing pytorch docker image:

    docker run --name langsam_pytorch --gpus all -it pytorch/pytorch:latest

## Inside the container:

-  after opening the container (I Recommend using vscode), clone the repository (using this fork):

    git clone https://github.com/kauevestena/lang-segment-anything.git

    cd lang-segment-anything

( or launch VScode on it)

# Troubleshooting "ImportError: libGL.so.1: cannot open shared object file: No such file or directory":

    apt update 
    apt install libgl1-mesa-glx -y
    apt install ffmpeg libsm6 libxext6 -y
    apt install libglib2.0-0 -y


Notice that generally inside the container you'll already have sudo already

# Install python requirements:

    python -m pip install -r requirements.txt

# Test if everything os working:

    python running_test.py


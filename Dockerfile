FROM pytorch/pytorch:latest

# prevent apt from hanging
ARG DEBIAN_FRONTEND=noninteractive

ENV HOME /workspace

WORKDIR $HOME

# dependency: lang-segment-anything
RUN apt update
RUN apt install -y git
RUN apt install libgl1-mesa-glx -y
RUN apt install libglib2.0-0 -y
RUN git clone https://github.com/kauevestena/lang-segment-anything.git
WORKDIR $HOME/lang-segment-anything
RUN pip install -e .
RUN python running_test.py
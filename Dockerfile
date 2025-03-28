# Use NVIDIA CUDA image with cuDNN support
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Set the DEBIAN_FRONTEND to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    git wget unzip \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install gdown to download files from Google Drive
RUN pip3 install --no-cache-dir gdown

# Create a user named david
RUN useradd -m david

# Set working directory and change ownership
RUN chown -R david:david /home/david/

RUN apt update && apt install -y nano

# Switch to non-root user
USER david

# Set the working directory
WORKDIR /home/david/app

# Copy project files into the container
COPY --chown=david . /home/david/app/

#RUN git clone https://github.com/Drq13112/PanopticSwiftNet.git

# Download pretrained models from Google Drive
RUN gdown --folder https://drive.google.com/drive/folders/1Z8PTS1PwF5ol9yFdLcop5z9_la1uy67q -O /home/david/app/PanopticSwiftNet/pretrained_models/

# Upgrade pip
RUN pip3 install --upgrade pip


RUN pip3 install --upgrade pip setuptools

# Install Detectron2 dependencies
RUN pip3 install torch torchvision torchaudio

# Install Detectron2 from the official source (latest version for CUDA 12)
#RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git'

RUN pip3 install git+https://github.com/cocodataset/panopticapi.git

RUN pip3 install cupy

# Extract dataset from a mounted directory (at runtime)
RUN mkdir -p /home/david/app/datasets/cityscapes


ENTRYPOINT ["/bin/bash"]
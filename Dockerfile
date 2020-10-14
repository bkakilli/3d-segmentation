FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

ENV USER=#SET_USER_NAME
ENV UID=#SET_UID
ENV GUI=#SET_GID

RUN apt update && \
    apt upgrade -y && \
    apt autoremove -y && \
    apt install -y \
        python3-pip \
        python3-dev \
        git \
        libgl1-mesa-glx

# Set Python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 2 && \
    update-alternatives --auto python && \
    update-alternatives --auto pip

# Install Pytorch (1.5.0)
RUN \
    pip install --upgrade --no-cache-dir \
        pip \
        pylint \
        h5py \
        tqdm \
        numpy \
        scipy \
        scikit-learn \
        scikit-image \
        open3d \
        torch==1.5.0 \
        torchvision==0.6 \
        tensorboard \
        git+https://gitlab.com/syr-svs/svs-tools.git

# Install Pytorch geometric
RUN pip install --no-cache-dir \
    torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html \
    torch-spline-conv==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html \
    torch-cluster==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html \
    torch-sparse==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html \
    torch-geometric

# Add local user
ENV HOME=/home/$USER
RUN useradd -s /bin/bash -u $UID -g $GID -m $HOME $USER && \
    usermod -aG sudo $USER
USER $USER

WORKDIR /seg
CMD bash
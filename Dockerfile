FROM pytorch/pytorch

RUN \
    pip install --upgrade --no-cache-dir \
        pip \
        h5py \
        tqdm \
        numpy \
        tensorboard \
        torch-scatter \
        torch-sparse \
        torch-cluster \
        torch-spline-conv \
        torch-geometric \
        open3d

WORKDIR /seg
CMD bash
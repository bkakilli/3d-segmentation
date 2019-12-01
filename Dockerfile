FROM pytorch/pytorch

RUN \
    pip install --upgrade \
        pip \
        h5py \
        tqdm \
        numpy \
        tensorboard

WORKDIR /seg
CMD bash
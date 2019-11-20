FROM pytorch/pytorch

RUN \
    pip install --upgrade \
        pip \
        h5py \
        tqdm \
        numpy

WORKDIR /seg
CMD bash
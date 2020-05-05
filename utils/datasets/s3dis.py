import os
import sys
import glob

import h5py
import numpy as np
from tqdm import tqdm
import torch.utils.data as data

# Add parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import augmentations


SPLITS = {
    1: {
        "train": ["Area_2", "Area_3", "Area_4", "Area_5", "Area_6"],
        "test": ["Area_1"],
        "val": ["Area_1"],
    },
    2: {
        "train": ["Area_1", "Area_3", "Area_4", "Area_5", "Area_6"],
        "test": ["Area_2"],
        "val": ["Area_2"],
    },
    3: {
        "train": ["Area_1", "Area_2", "Area_4", "Area_5", "Area_6"],
        "test": ["Area_3"],
        "val": ["Area_3"],
    },
    4: {
        "train": ["Area_1", "Area_2", "Area_3", "Area_5", "Area_6"],
        "test": ["Area_4"],
        "val": ["Area_4"],
    },
    5: {
        "train": ["Area_1", "Area_2", "Area_3", "Area_4", "Area_6"],
        "test": ["Area_5"],
        "val": ["Area_5"],
    },
    6: {
        "train": ["Area_1", "Area_2", "Area_3", "Area_4", "Area_5"],
        "test": ["Area_6"],
        "val": ["Area_6"],
    },
}

def read_file_items(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def load_dataset(root):
    all_files = read_file_items(os.path.join(root, 'indoor3d_sem_seg_hdf5_data/all_files.txt'))
    all_files = [os.path.join(root, f) for f in all_files]

    # Load ALL data
    data_batch_list = []
    label_batch_list = []
    for h5_filename in tqdm(all_files, ncols=100, desc="Loading dataset into RAM"):
        data_batch, label_batch = loadDataFile(h5_filename)
        data_batch_list.append(data_batch)
        label_batch_list.append(label_batch)
    data_batches = np.concatenate(data_batch_list, 0)
    label_batches = np.concatenate(label_batch_list, 0)

    return data_batches, label_batches


def get_indices(root, areas):
    room_filelist = read_file_items(os.path.join(root,'indoor3d_sem_seg_hdf5_data/room_filelist.txt'))
    indices = []
    for area in areas:
        indices += [i for i, bn in enumerate(room_filelist) if area in bn]

    return indices


class S3DISDataset(data.Dataset):
    def __init__(self, root="data", split="test", augmentation=False, preload=None):
        
        # Load the entire dataset from disk to memory if it is not already provided
        loaded = load_dataset(root) if preload is None else preload
        self.data, self.labels = loaded

        if isinstance(split, str):
            # Default test data is Room #5
            areas = SPLITS[5][split]
        elif isinstance(split, list):
            areas = split
        else:
            raise ValueError("Unsupported split type.")

        self.data_indices = get_indices(root, areas)

        self.augmentation = augmentation   

    def augment_data(self, data, label):
        
        num_points = len(data)
        batch_data = np.vstack((data[:, :3], data[:, -3:]))[np.newaxis, ...]

        # batch_data, label, _ = augmentations.shuffle_data(batch_data, label)
        batch_data = augmentations.rotate_point_cloud(batch_data)
        batch_data = augmentations.shift_point_cloud(batch_data)
        batch_data = augmentations.random_scale_point_cloud(batch_data)
        batch_data = augmentations.jitter_point_cloud(batch_data)

        data = np.hstack((batch_data[0][:num_points], data[:, 3:6], batch_data[0][num_points:]))

        return data, label


    def __getitem__(self, i):

        index = self.data_indices[i]
        data, labels = self.data[index], self.labels[index]

        if self.augmentation:
            data, labels = self.augment_data(data, labels)

        # Make it channels first
        data = np.swapaxes(data, 0, 1).astype(np.float32)

        return data, labels.astype(np.int64)

    def __len__(self):
        return len(self.data_indices)

def get_sets(data_folder, split_id=None, training_augmentation=True):
    """Return hooks to S3DIS dataset train, validation and tests sets.
    """

    train_set = S3DISDataset(data_folder, split=SPLITS[split_id]['train'], augmentation=training_augmentation)

    # Use the same loaded data
    preload = (train_set.data, train_set.labels)
    valid_set = S3DISDataset(data_folder, split=SPLITS[split_id]['val'], preload=preload)
    test_set = S3DISDataset(data_folder, split=SPLITS[split_id]['test'], preload=preload)

    return train_set, valid_set, test_set


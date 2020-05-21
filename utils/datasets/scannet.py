import os
import sys

import numpy as np
import torch.utils.data as torch_data
from tqdm import tqdm

# Add parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import augmentations


def take_cell_sample(capture_data, sample_at=None, num_points=8192, dims=None, method="random", min_N=256):

    if dims is None:
        dims = [1.0, 1.0, 99.0]
    dims = np.asarray(dims)
    
    if method is "random":

        def random_sample():
            min_coords = capture_data[:, :3].min(axis=0)
            max_coords = capture_data[:, :3].max(axis=0)
            span = np.array(max_coords - min_coords)

            # Origin
            o = min_coords + np.random.rand(3)*span - dims/2

            mask = np.logical_and.reduce((
                capture_data[:, 0] > o[0],
                capture_data[:, 1] > o[1],
                capture_data[:, 0] <= o[0]+dims[0],
                capture_data[:, 1] <= o[1]+dims[1],
            ))

            return capture_data[mask]

        # Try sampling until we have sufficient points
        sampled = random_sample()
        while len(sampled) < min_N:
            sampled = random_sample()

        # Sample additional points (copies) if total is less then desired
        if len(sampled) < num_points:
            additional_samples = sampled[np.random.choice(len(sampled), num_points-len(sampled), replace=True)]
            sampled = np.vstack((sampled, additional_samples))
        else:
            sampled = np.random.permutation(sampled)[:num_points]
        
        return sampled


def label_correct(l):
    order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39, 20]
    return order.index(l)

class ScanNetDataset(torch_data.Dataset):

    def __init__(self, root="data", split="train", num_points=4096, augmentation=False):
        
        self.loaded = np.load(os.path.join(root, "preloaded.npz"))
        self.captures = [f for f in self.loaded.files if split in f]
        self.num_points = num_points
        self.augmentation = augmentation 

    def augment_data(self, data, label):
        
        batch_data = data[np.newaxis, :, :3]

        batch_data = augmentations.rotate_point_cloud(batch_data)
        batch_data = augmentations.shift_point_cloud(batch_data)
        batch_data = augmentations.random_scale_point_cloud(batch_data)
        batch_data = augmentations.jitter_point_cloud(batch_data)

        data = np.hstack((batch_data[0], data[:, 3:6]))

        return data, label


    def __getitem__(self, i):

        capture_data = self.loaded[self.captures[i]]
        
        sampled = take_cell_sample(capture_data, num_points=self.num_points, min_N=512)
        data, labels = sampled[:, :6], sampled[:, 6]

        # TEMPORARY CORRECT LABELS
        labels[labels == -1] = 20
        label_correct_vectorize = np.vectorize(label_correct)
        labels = label_correct_vectorize(labels)

        if self.augmentation:
            data, labels = self.augment_data(data, labels)

        # Make it channels first
        data = np.swapaxes(data, 0, 1).astype(np.float32)

        return data, labels.astype(np.int64)

    def __len__(self):
        return len(self.captures)


def get_sets(data_folder, split_id=None, training_augmentation=True):
    """Return hooks to ScanNet dataset train, validation and tests sets.
    """

    train_set = ScanNetDataset(data_folder, split='train', augmentation=training_augmentation)
    valid_set = ScanNetDataset(data_folder, split='val')
    test_set = ScanNetDataset(data_folder, split='test')

    # train_set = ScanNetDataset(data_folder, split='train', augmentation=training_augmentation)

    # train_ratio = 0.9
    # train_split = int(len(train_set)*train_ratio)
    # valid_split = len(train_set) - train_split
    
    # train_set, valid_set = torch_data.random_split(train_set, (train_split, valid_split))
    # test_set = ScanNetDataset(data_folder, split='test') 

    return train_set, valid_set, test_set


def test():
    print("loading dataset")
    dataloader = ScanNetDataset('/data1/datasets/scannet_preprocessed', split='train') 
    print("reading data")
    inpt, oupt = dataloader[3]


            
if __name__== '__main__':
    test()

import os
import sys

import numpy as np
import torch.utils.data as torch_data
from tqdm import tqdm

# Add parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import augmentations


def take_random_cell_sample(capture_data, sample_at=None, num_points=8192, dims=None, method="random", min_N=256):

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
        
    # Normalize
    sampled[:, :3] -= sampled[:, :3].mean(axis=0, keepdims=True)

    return sampled


class ScanNetDataset(torch_data.Dataset):

    def __init__(self, root="data", split="train", num_points=4096, augmentation=False):
        
        self.loaded = np.load(os.path.join(root, "preloaded_512.npz"))
        # self.captures = [f for f in self.loaded.files if f.startswith(split+"/captures/")]
        self.cells = [f for f in self.loaded.files if f.startswith(split+"/cells/")]
        self.num_points = num_points
        self.augmentation = augmentation

        if split is "test":
            self.labelweights = np.ones(21, dtype=np.float32)
        else:
            labelweights = self.loaded[split+"/count"]
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = (1/np.log(1.2+labelweights)).astype(np.float32)

        self.num_labels = 21    # Including negative class
        np.random.shuffle(self.cells)
        self.cells = self.cells[:int(len(self.cells)*0.1)]

        self.caching_enabled = True
        self.cache = {}
        

    def augment_data(self, data, label):
        
        batch_data = data[np.newaxis, :, :3]

        batch_data = augmentations.rotate_point_cloud(batch_data, rotation_axis='z')
        # batch_data = augmentations.shift_point_cloud(batch_data)
        batch_data = augmentations.random_scale_point_cloud(batch_data)
        # batch_data = augmentations.jitter_point_cloud(batch_data)

        data = np.hstack((batch_data[0], data[:, 3:6]))

        return data, label

    def load_from_cache(self, address):

        if self.caching_enabled:
            if address in self.cache:
                data = self.cache[address]
            else:
                data = self.loaded[address]
                self.cache[address] = data
        else:
            data = self.loaded[address]

        return data


    def __getitem__(self, i):

        cell_path = self.cells[i]
        cell_indices = self.load_from_cache(cell_path)

        split, _, cell_name = cell_path.split("/")  # train / cells / scene0119_00_cell_003.npy
        capture_path = "%s/captures/%s" % (split, cell_name[:12])

        capture_data = self.load_from_cache(capture_path)
        
        sampled = capture_data[cell_indices]
        # Normalize
        sampled[:, :3] -= sampled[:, :3].mean(axis=0, keepdims=True)

        # sampled = take_random_cell_sample(capture_data, num_points=self.num_points, min_N=512)
        data, labels = sampled[:, :6], sampled[:, 6]

        if self.augmentation:
            data, labels = self.augment_data(data, labels)

        # Make it channels first
        data = np.swapaxes(data, 0, 1).astype(np.float32)

        return data, labels.astype(np.int64)

    def __len__(self):
        return len(self.cells)


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

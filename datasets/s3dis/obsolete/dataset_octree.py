import os
import pickle

import numpy as np
import torch

from utils import augmentations, octree_utils, misc


def custom_collate_fn(batch):

    data = np.asarray([item[0] for item in batch]).astype(np.float32)
    labels = np.asarray([item[1] for item in batch]).astype(np.int64)
    meta = np.asarray([item[2] for item in batch])

    data = misc.to_tensor(data)
    labels = misc.to_tensor(labels)
    meta = misc.to_tensor(meta)

    return [data, labels, meta]

class Dataset(torch.utils.data.Dataset):

    def __init__(self, root=None, split="train", crossval_id=0, num_points=2**18, augmentation=False, **kwargs):
        
        if root is None:
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))

        with open(os.path.join(root, "meta.pkl"), "rb") as f_handler:
            self.meta = pickle.load(f_handler)

        self.room_paths = [os.path.join(root, r) for r in self.meta[crossval_id][split]["paths"]]

        # Temporary filter out auditorium
        self.room_paths = [p for p in self.room_paths if "auditorium" not in p]

        self.num_points = num_points
        self.augmentation = augmentation if split == "train" else False

        self.num_labels = len(self.meta["labels"])    # Including negative class
        self.levels = [5, 3]

        if split is "test":
            self.labelweights = np.ones(self.num_labels, dtype=np.float32)
        else:
            labelweights = self.meta[crossval_id][split]["count"]
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = (1/np.log(1.2+labelweights)).astype(np.float32)

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
                data = np.load(self.room_paths[address])
                self.cache[address] = data
        else:
            data = np.load(self.room_paths[address])

        return data

    def normalize(self, pc):
        # pc -= pc.mean(axis=0, keepdims=True)
        # pc /= np.max(pc.max(axis=0) - pc.min(axis=0))
        pc -= pc.min(axis=0, keepdims=True)
        pc /= np.max(pc.max(axis=0) - pc.min(axis=0))
        return pc

    def __getitem__(self, i):

        scene = self.load_from_cache(i).astype(np.float32)

        # Normalize
        scene[:, :3] = self.normalize(scene[:, :3])

        # Get target indices
        indices = np.arange(len(scene))
        if len(scene) < self.num_points:
            indices = np.append(indices, np.random.choice(len(scene), self.num_points-len(scene)))
        np.random.shuffle(indices)
        indices = indices[:self.num_points] if self.num_points > 0 else indices

        data, labels = scene[indices, :6], scene[indices, 6]
        
        meta = octree_utils.make_groups(data, self.levels, 0.0)

        # if self.augmentation:
        #     data, labels = self.augment_data(data, labels)

        # Make it channels first
        data = np.swapaxes(data, 0, 1)

        return data, labels, meta

    def __len__(self):
        return len(self.room_paths)


def test():

    from svstools import visualization as vis
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    print("Reading data")
    for split in ["train", "val", "test"]:
        dataset = Dataset(split=split, crossval_id=5)
        dl = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)
        for batch_cpu in tqdm(dl, desc=split):
            batch = misc.move_to(batch_cpu, torch.device("cuda:0"))

    return


            
if __name__== '__main__':
    test()

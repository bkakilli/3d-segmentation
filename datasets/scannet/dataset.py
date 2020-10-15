import os
import sys
import glob

import numpy as np
import torch
import torch.utils.data as torch_data

# Add parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import augmentations
import octree_utils
import misc


def custom_collate_fn(batch):

    data = np.asarray([item[0] for item in batch]).astype(np.float32)
    groups = np.asarray([item[1] for item in batch])
    labels = np.asarray([item[2] for item in batch]).astype(np.int64)

    data = misc.to_tensor(data)
    groups = misc.to_tensor(groups)
    labels = misc.to_tensor(labels)

    return [data, groups, labels]

class Dataset(torch_data.Dataset):

    def __init__(self, root=None, split="train", num_points=2**16, augmentation=False, **kwargs):
        
        if root is None:
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
        
        self.split_path = os.path.join(root, "preloaded_512", split)
        self.captures = glob.glob(self.split_path + "/captures/*.npy")
        self.num_points = num_points
        self.augmentation = augmentation if split == "train" else False

        self.num_labels = 21    # Including negative class
        self.levels = [5, 3, 1]

        if split is "test":
            self.labelweights = np.ones(self.num_labels, dtype=np.float32)
        else:
            labelweights = np.load(self.split_path + "/count.npy")
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
                data = np.load(self.captures[address])
                self.cache[address] = data
        else:
            data = np.load(self.captures[address])

        return data

    def normalize(self, pc):
        # pc -= pc.mean(axis=0, keepdims=True)
        # pc /= np.max(pc.max(axis=0) - pc.min(axis=0))
        pc -= pc.min(axis=0, keepdims=True)
        pc /= np.max(pc.max(axis=0) - pc.min(axis=0))
        return pc

    def __getitem__(self, i):

        scene = self.load_from_cache(i)

        # Normalize
        scene[:, :3] = self.normalize(scene[:, :3])

        # Get target indices
        indices = np.arange(len(scene))
        if len(scene) < self.num_points:
            indices = np.append(indices, np.random.choice(len(scene), self.num_points-len(scene)))
        np.random.shuffle(indices)
        indices = indices[:self.num_points]

        data, labels = scene[indices, :6], scene[indices, 6]
        
        groups = octree_utils.make_groups(data, self.levels, 0.0)

        # if self.augmentation:
        #     data, labels = self.augment_data(data, labels)

        # Make it channels first
        data = np.swapaxes(data, 0, 1)

        return data, groups, labels
        # return {"data": data, "groups": groups, "labels": labels.astype(np.int64)}

    def __len__(self):
        return len(self.captures)


def test():

    from svstools import visualization as vis

    print("loading dataset")
    dataloader = Dataset('data/scannet', split='train') 

    print("reading data")
    data, groups, label = dataloader[3]

    return


            
if __name__== '__main__':
    test()

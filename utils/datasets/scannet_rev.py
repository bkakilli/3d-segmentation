import os
import sys
import glob

import numpy as np
import torch
import torch.utils.data as torch_data
from tqdm import tqdm

from svstools import pc_utils

# Add parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import augmentations


def to_tensor(obj):

    if torch.is_tensor(obj):
        return obj

    if isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = to_tensor(v)
        return res

    if isinstance(obj, list):
        res = []
        for v in obj:
            res.append(to_tensor(v))
        return res

    if isinstance(obj, tuple):
        res = []
        for v in obj:
            res.append(to_tensor(v))
        return tuple(res)

    if isinstance(obj, np.ndarray):
        if obj.dtype in [np.float64, np.float32, np.float16, np.int64, np.int32, np.int16, np.int8, np.uint8, np.bool]:
            return torch.from_numpy(obj)
        else:
            return (to_tensor(obj.tolist()))

    raise TypeError("Invalid type for to_tensor")

def walk_octree(tree, size_expand):

    child_map = np.array([
        [0,0,0],
        [0,0,1],
        [0,1,0],
        [0,1,1],
        [1,0,0],
        [1,0,1],
        [1,1,0],
        [1,1,1],
    ])

    child_map = np.fliplr(child_map)

    def recursive_walk(node, size, origin):

        leafs = []
        for i, child in enumerate(node.children):
            if child is None:
                continue

            child_size = size/2
            child_origin = origin + child_map[i]*child_size
            if isinstance(child, pc_utils.o3d.geometry.OctreeColorLeafNode):
                leafs.append([child_origin, child_size])
            else:
                leafs += recursive_walk(child, child_size, child_origin)

        return leafs

    root = tree.root_node
    size = np.round(tree.size-size_expand, decimals=5)
    origin = np.round(tree.origin, decimals=5)

    return recursive_walk(root, size, origin)
    

def make_octree_group(cloud, octree):

    leafs = walk_octree(octree, size_expand=0.0)
    origins, sizes = [], []
    for leaf in leafs:
        origins.append(leaf[0])
        sizes.append(leaf[1])

    sizes = np.array(sizes).reshape(-1, 1, 1)
    origins = np.reshape(origins, (-1, 3, 1))
    bboxes = np.concatenate((origins, origins+sizes+1e-8), axis=-1).reshape(-1, 6)

    groups = []

    for bbox in bboxes:
        indices = pc_utils.crop_bbox(cloud, bbox)
        if len(indices) > 1:
            groups.append(indices)

    seen = np.zeros((len(cloud),), dtype=np.int)
    for g in groups:
        seen[g] += 1
    
    dublicates = np.where(seen > 1)[0]
    unseen = np.where(seen == 0)[0]

    return groups

def make_groups(pc, levels, size_expand=0.01):

    pc = pc.copy()

    octrees = {}
    groups = {}
    for level in levels:
        pcd = pc_utils.points2PointCloud(pc)
        if level == 1:
            octree_group = [np.arange(len(pc))]
        else:
            octrees[level] = pc_utils.o3d.geometry.Octree(max_depth=level)
            octrees[level].convert_from_point_cloud(pcd, size_expand)

            octree_group = make_octree_group(pc, octrees[level])

        means_of_groups = [pc[g_i].mean(axis=0, keepdims=True) for g_i in octree_group]
        pc = np.row_stack(means_of_groups)

        groups[level] = (np.transpose(pc, (1, 0)), octree_group)

    return groups

def custom_collate_fn(batch):

    data = np.asarray([item[0] for item in batch]).astype(np.float32)
    groups = np.asarray([item[1] for item in batch])
    labels = np.asarray([item[2] for item in batch]).astype(np.int64)

    data = to_tensor(data)
    groups = to_tensor(groups)
    labels = to_tensor(labels)

    return [data, groups, labels]

class ScanNetDataset(torch_data.Dataset):

    def __init__(self, root="data", split="train", num_points=2**16, augmentation=False):
        
        self.split_path = os.path.join(root, "preloaded_512", split)
        self.captures = glob.glob(self.split_path + "/captures/*.npy")
        self.num_points = num_points
        self.augmentation = augmentation

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


    def __getitem__(self, i):

        scene = self.load_from_cache(i)

        # Normalize
        scene[:, :3] -= scene[:, :3].mean(axis=0, keepdims=True)

        # Get target indices
        indices = np.arange(len(scene))
        if len(scene) < self.num_points:
            indices = np.append(indices, np.random.choice(len(scene), self.num_points-len(scene)))
        np.random.shuffle(indices)
        indices = indices[:self.num_points]

        data, labels = scene[indices, :6], scene[indices, 6]
        
        groups = make_groups(data, self.levels, 0.0)

        # if self.augmentation:
        #     data, labels = self.augment_data(data, labels)

        # Make it channels first
        data = np.swapaxes(data, 0, 1)

        return data, groups, labels
        # return {"data": data, "groups": groups, "labels": labels.astype(np.int64)}

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

    from svstools import visualization as vis

    print("loading dataset")
    dataloader = ScanNetDataset('data/scannet', split='train') 

    print("reading data")
    data, groups, label = dataloader[3]

    return


            
if __name__== '__main__':
    test()

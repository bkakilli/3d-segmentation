import os
import pickle

import numpy as np
import torch

from utils import augmentations, pc_utils, misc


# def custom_collate_fn(batch):

#     data = np.asarray([item[0] for item in batch]).astype(np.float32)
#     labels = np.asarray([item[1] for item in batch]).astype(np.int64)
#     meta = np.asarray([item[2] for item in batch])

#     data = misc.to_tensor(data)
#     labels = misc.to_tensor(labels)
#     meta = misc.to_tensor(meta)

#     return [data, labels, meta]
custom_collate_fn = None

class Dataset(torch.utils.data.Dataset):

    def __init__(self, root=None, split="train", crossval_id=5, num_points=2**18, augmentation=False, **kwargs):
        
        if root is None:
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))

        self.root = root
        with open(os.path.join(self.root, "meta.pkl"), "rb") as f_handler:
            self.meta = pickle.load(f_handler)

        # self.room_paths = [os.path.join(root, r) for r in self.meta[crossval_id][split]["paths"]]

        # # Temporary filter out auditorium
        # self.room_paths = [p for p in self.room_paths if "auditorium" not in p]

        self.num_points = num_points
        self.augmentation = augmentation if split == "train" else False

        self.num_labels = len(self.meta["labels"])    # Including negative class

        self.meta["block_size"] = 0.5
        # self.make_groups(self.meta[crossval_id][split]["blocks"], K=16)
        self.groups = []
        for area in self.meta[crossval_id][split]["areas"]:
            self.groups += self.meta["groups"][area]

        self.neig_K = 8

        if split is "test":
            self.labelweights = np.ones(self.num_labels, dtype=np.float32)
        else:
            labelweights = self.meta[crossval_id][split]["count"]
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = (1/np.log(1.2+labelweights)).astype(np.float32)

        self.caching_enabled = True
        self.cache = {}
    
    def make_groups(self, block_indices, K):
        self.room_names = {}
        self.area_names = {}
        self.groups = []
        for b in block_indices:
            block = self.meta["blocks"][b]
            
            # Memory friendly room name retrieval
            room_name_key = 0
            while room_name_key in self.room_names:
                if self.room_names[room_name_key] == block["room_name"]:
                    break
                room_name_key += 1
            self.room_names[room_name_key] = block["room_name"]
            
            # Memory friendly area name retrieval
            area_name_key = 0
            while area_name_key in self.area_names:
                if self.area_names[area_name_key] == block["area"]:
                    break
                area_name_key += 1
            self.area_names[area_name_key] = block["area"]

            kdtree = pc_utils.KDTree(block["coordinates"])

            for c, g in zip(block["coordinates"], block["groups"]):
                neighborhood = []
                for neig in kdtree.query([c], k=K, return_distance=False)[0]:
                    neighborhood.append({
                        "indices": block["groups"][neig],
                        "coordinate": block["coordinates"][neig],
                    })
                group = {
                    "area": area_name_key,
                    "room_name": room_name_key,
                    "neighborhood": neighborhood, # self as index=0
                }

                self.groups.append(group)
            

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
                data = np.load(address)
                self.cache[address] = data
        else:
            data = np.load(address)

        return data

    def normalize(self, pc):
        # pc -= pc.mean(axis=0, keepdims=True)
        # pc /= np.max(pc.max(axis=0) - pc.min(axis=0))
        pc -= pc.min(axis=0, keepdims=True)
        pc /= np.max(pc.max(axis=0) - pc.min(axis=0))
        return pc

    def __getitem__(self, i):
        
        group = self.groups[i]
        address = os.path.join(self.root, self.meta["paths"][group["path"]])
        room_cloud = self.load_from_cache(address).astype(np.float32)
        
        neighborhood = []
        for c, g in zip(group["coordinates"][:self.neig_K], group["neighborhood"][:self.neig_K]):
            group_points = room_cloud[self.meta["indices"][g], :6]
            group_points[:, :3] -= [c + self.meta["block_size"]/2]
            
            # if self.augmentation:
            #     data, labels = self.augment_data(data, labels)

            neighborhood.append(group_points)

        data = np.asarray(neighborhood)
        labels = room_cloud[self.meta["indices"][group["neighborhood"][0]], 6].astype(int)

        # data.shape is (Neig, N, 6)
        # labels.shape is (N,)

        # Make it channels first
        data = np.transpose(data, (2,0,1))

        return data, labels

    def __len__(self):
        return len(self.groups)


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

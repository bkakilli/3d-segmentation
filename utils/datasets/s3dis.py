import os
import sys
import glob

import numpy as np
import torch.utils.data as data

# Add parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import augmentations

class S3DISDataset(data.Dataset):
    def __init__(self, root="data/Stanford3d_batch_version", split="test", augmentation=False):
        self.root = root
        self.cls_list=['clutter', 'ceiling', 'floor', 'wall', 'beam', 'column',
                    'door', 'window', 'table', 'chair', 'sofa', 'bookcase', 'board']

        if isinstance(split, str):
            split_areas = {
                "train": ["Area_1", "Area_2", "Area_3", "Area_4", "Area_6"],
                "test": ["Area_5"],
                "val": ["Area_5"],
            }
            areas = split_areas[split]
        elif isinstance(split, list):
            areas = split
        else:
            raise ValueError("Unsupported split type.")

        self.augmentation = augmentation

        self.room_list = self.create_room_list(areas)
        self.batch_list = self.create_batch_list()


    def create_room_list(self, area_list):
        room_list = []
        for area in area_list:
            area_pattern = os.path.join(self.root, area, "[!.]*")
            room_list += [room for room in glob.glob(area_pattern)]

        return room_list


    def create_batch_list(self):
        batch_list=[]
        for room in self.room_list:
            room_batch_path=os.path.join(room, 'Batch_Folder')
            room_batch_list=os.listdir(room_batch_path)[0:100]
            for batch_data in room_batch_list:
                batch_data_path=os.path.join(room_batch_path,batch_data)
                batch_list.append(batch_data_path)
        return batch_list

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


    def __getitem__(self,batch_index):
        txt_file = self.batch_list[batch_index]
        data = np.loadtxt(txt_file)
        inpt = data[:, 0:9]
        label = data[:, -1].astype(np.int64)

        if self.augmentation:
            inpt, label = self.augment_data(inpt, label)

        # Make it channels first
        inpt = np.swapaxes(inpt, 0, 1).astype(np.float32)

        return inpt, label

    def __len__(self):
        return len(self.batch_list)


def get_sets(data_folder, training_augmentation=True):
    """Return hooks to S3DIS dataset train, validation and tests sets.
    """

    train_set = S3DISDataset(data_folder, split='train', augmentation=training_augmentation)
    valid_set = S3DISDataset(data_folder, split='val')
    test_set = S3DISDataset(data_folder, split='test')

    return train_set, valid_set, test_set

def test():
    from svstools import visualization as vis
    datafolder='/home/burak/datasets/Stanford3d_batch_version'
    t, _, _ = get_sets(datafolder, training_augmentation=False)

    for i in range(10,20):
        X, y = t[i]
        X, y = X.numpy(), y.numpy()

        pcd = vis.paint_colormap(X[-3:].T, y)
        vis.show_pointcloud(pcd)

    return

if __name__=='__main__':
    test()

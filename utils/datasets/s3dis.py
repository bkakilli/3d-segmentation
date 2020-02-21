import numpy as np
import torch
import os
import glob
import open3d as o3d
import torch.utils.data as data

class S3DISDataset(data.Dataset):
    def __init__(self, root, split, path_prefix=None, augmentation=False):
        if path_prefix is not None:
            root = os.path.join(path_prefix, root)
        self.root = root
        self.cls_list=['clutter', 'ceiling', 'floor', 'wall', 'beam', 'column',
                    'door', 'window', 'table', 'chair', 'sofa', 'bookcase', 'board']

        split_areas = {
            "train": ["Area_1", "Area_2", "Area_3", "Area_4", "Area_6"],
            "test": ["Area_5"],
            "val": ["Area_5"]
        }

        self.room_list = self.create_room_list(split_areas[split])
        self.batch_list=self.create_batch_list()


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


    def __getitem__(self,batch_index):
        txt_file=self.batch_list[batch_index]
        data=np.loadtxt(txt_file)
        inpt=torch.FloatTensor(data[:,0:9]).permute(1,0)
        label=torch.LongTensor(data[:,-1])
        return inpt,label

    def __len__(self):
        return len(self.batch_list)


def get_sets(data_folder, path_prefix=None, training_augmentation=True):
    """Return hooks to S3DIS dataset train, validation and tests sets.
    """

    train_set = S3DISDataset(data_folder, split='train',path_prefix=path_prefix, augmentation=training_augmentation)
    valid_set = S3DISDataset(data_folder, split='val', path_prefix=path_prefix)
    test_set = S3DISDataset(data_folder, split='test', path_prefix=path_prefix)

    return train_set, valid_set, test_set


if __name__=='__main__':
    datafolder='/data1/datasets/Stanford3dDataset_v1.2'
    train,va,test=get_sets(datafolder,path_prefix=None)
    print(train[20][0].shape)

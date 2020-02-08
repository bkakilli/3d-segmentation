import torch
import os
import glob
import numpy as np
import open3d as o3d

np.random.seed(0)

class S3DISDataset(torch.utils.data.Dataset):

    def __init__(self, root='Stanford3dDataset_v1.2', split='train', path_prefix=None, augmentation=False):
        # raise NotImplementedError()
        if path_prefix is not None:
            root = os.path.join(path_prefix, root)
        self.root = root
        self.cls_list = ['clutter', 'ceiling', 'floor', 'wall', 'beam', 'column',
                    'door', 'window', 'table', 'chair', 'sofa', 'bookcase', 'board']

        split_areas = {
            "train": ["Area_1", "Area_2", "Area_3", "Area_4", "Area_6"],
            "test": ["Area_5"],
            "val": ["Area_5"]
        }
        self.room_list = self.create_room_list(split_areas[split])



    def create_room_list(self, area_list):
        room_path_list = []
        for area in area_list:
            rooms_pattern = os.path.join(self.root, area, "[!.]*")
            room_list = glob.glob(rooms_pattern)
            area_rooms = [room for room in room_list]
            room_path_list += area_rooms

        return room_path_list


    def create_inpt_data(self,room_index):
        point_list = []
        output_label = []
        room_path = self.room_list[room_index]
        annotation_path = os.path.join(room_path, 'Annotations')
        label_list = os.listdir(annotation_path)
        for label_file in label_list:
            if label_file in ['.DS_Store', "Icon"]:
                continue
            label_name = label_file.split('_')[0]
            if label_name=='stairs':
                continue
            label_index = self.cls_list.index(label_name)
            label_file = os.path.join(annotation_path, label_file)
            file = open(label_file).read()
            file = file.split('\n')
            # output_label.append([label_index]*len(file))
            for point in file:
                point = point.split(' ')
                if len(point) == 6:
                    point = point[:3]
                    point_value = [np.float(i) for i in point]
                    point_list.append(point_value)
                    output_label.append(label_index)
                else:
                    continue
        point_list = np.array(point_list).reshape(-1, 3)
        point_list=torch.FloatTensor(point_list)
        output_label=torch.LongTensor(output_label)
        return point_list, output_label



    def visula_room(self,room_index):
        point_list,output_label=self.create_inpt_data(room_index)
        point_list=point_list.numpy()
        output_label=output_label.numpy()
        color_label = np.array([[0, 255, 0],
                           [0, 0, 255],
                           [0, 255, 255],
                           [255, 255, 0],
                           [255, 0, 255],
                           [100, 100, 255],
                           [200, 200, 100],
                           [170, 120, 200],
                           [255, 0, 0],
                           [200, 100, 100],
                           [10, 200, 100],
                           [200, 200, 200],
                           [50, 50, 50]])     
        color_label = color_label/255
        color_vector=color_label[output_label]
        vis_points=o3d.geometry.PointCloud()
        vis_points.points=o3d.utility.Vector3dVector(point_list)
        vis_points.colors=o3d.utility.Vector3dVector(color_vector)
        o3d.visualization.draw_geometries([vis_points])



    def __len__(self):
        lens=len(self.room_list)
        return lens


    def __getitem__(self,index):
        point_list,output_label=self.create_inpt_data(index)
        len_inpt=output_label.shape[0]
        indi=np.arange(len_inpt)
        np.random.shuffle(indi)
        indi=indi[:5000]
        inpt=point_list[indi].permute(1,0)
        label=output_label[indi]
        return inpt,label

    # @staticmethod
    # def get_transforms():
    #     return None

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
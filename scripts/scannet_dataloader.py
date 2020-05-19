import numpy as np
import torch
import os
import torch.utils.data as torch_data

np.random.seed(0)

class ScanNetDataset(torch_data.Dataset):
    def __init__(self, root, split, augmentation=None):
        self.cls_list= [
        'floor', 
        'wall', 
        'cabinet', 
        'bed', 
        'chair', 
        'sofa', 
        'table', 
        'door', 
        'window', 
        'bookshelf', 
        'picture', 
        'counter', 
        'desk', 
        'curtain', 
        'refrigerator', 
        'bathtub', 
        'shower curtain', 
        'toilet', 
        'sink', 
        'otherprop']
        
        self.root = root
        self.split = split
        
        self.batch_list=self.create_batch_list()        
        
        
        
        
    def create_batch_list(self):
        if self.split == 'train':
            folder = 'scans'
        elif self.split == 'test':
            folder = 'scans_test'
        
        all_batch_list = []
        room_path = os.path.join(self.root, folder)
        room_list = os.listdir(room_path)
        for room in room_list:
            batch_folder_path = os.path.join(room_path, room, 'batch_folder')
            
            # Randomly pick up 5 batch for each room
            batch_list = os.listdir(batch_folder_path)
            picked_value = 5
            if picked_value > len(batch_list):
                picked_value = len(batch_list)

            batch_indice = np.random.choice(np.arange(len(batch_list)), picked_value, replace=False)
            for indice in batch_indice:
                picked_batch = os.path.join(batch_folder_path, batch_list[indice])
                all_batch_list.append(picked_batch)
        
        return all_batch_list
    
    def __len__(self):
        return len(self.batch_list)
    
    def __getitem__(self,batch_index):
        txt_file = self.batch_list[batch_index]
        data = np.loadtxt(txt_file)
        inpt = torch.FloatTensor(data[:,0:9])
        label = torch.LongTensor(data[:,-1])
        return inpt, label
    
def get_sets(data_folder, training_augmentation=True):
    """Return hooks to ScanNet dataset train, validation and tests sets.
    """

    train_set = ScanNetDataset(data_folder, split='train', augmentation=training_augmentation)

    train_ratio = 0.9
    train_split = int(len(train_set)*train_ratio)
    valid_split = len(train_set) - train_split
    
    train_set, valid_set = torch_data.random_split(train_set, (train_split, valid_split))
    test_set = ScanNetDataset(data_folder, split='test') 

    return train_set


def test():
    dataloader = ScanNetDataset('/data1/datasets/ScanNet/DATA', split='train') 
    inpt, oupt = dataloader[3] 


            
if __name__== '__main__':
    test()
    
    
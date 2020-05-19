import numpy as np
import torch
import os
import torch.utils.data as data

np.random.seed(0)

class ScanNetDataset(data.Dataset):
    def __init__(self,root,split,path_prefix=None,augmentation=None):
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
        
        self.root=root
        self.split=split
        
        self.batch_list=self.create_batch_list()        
        
        
        
        
    def create_batch_list(self):
        if self.split=='train':
            folder='scans'
        elif self.split=='test':
            folder='scans_test'
        
        all_batch_list=[]
        room_path=os.path.join(self.root,folder)
        room_list=os.listdir(room_path)
        for room in room_list:
            batch_folder_path=os.path.join(room_path,room,'batch_folder')
            
            #randomly pick up 5 batch for each room
            batch_list=os.listdir(batch_folder_path)
            picked_value=5
            if picked_value>len(batch_list):
                picked_value=len(batch_list)

            batch_indice=np.random.choice(np.arange(len(batch_list)),picked_value,replace=False)
            for indice in batch_indice:
                picked_batch=os.path.join(batch_folder_path,batch_list[indice])
                all_batch_list.append(picked_batch)
        
        return all_batch_list
    
    def __len__(self):
        return len(self.batch_list)
    
    def __getitem__(self,batch_index):
        txt_file=self.batch_list[batch_index]
        data=np.loadtxt(txt_file)
        inpt=torch.FloatTensor(data[:,0:9])
        label=torch.LongTensor(data[:,-1])
        return inpt,label
    
def get_sets(data_folder, path_prefix=None, training_augmentation=True):
    """Return hooks to S3DIS dataset train, validation and tests sets.
    """

    train_set = ScanNetDataset(data_folder, split='train',path_prefix=path_prefix, augmentation=training_augmentation)
    # valid_set = S3DISDataset(data_folder, split='val', path_prefix=path_prefix)
    # test_set = S3DISDataset(data_folder, split='test', path_prefix=path_prefix)

    return train_set



            
if __name__=='__main__':
    dataloader=ScanNetDataset('/data1/datasets/ScanNet/DATA',split='train') 
    inpt,oupt=dataloader[3] 
    
    
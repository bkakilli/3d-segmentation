import torch

class ScanNetDataset(torch.utils.data.Dataset):

    def __init__(self, data_folder="data/scannet", split="test", augmentation=False):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, i):
        raise NotImplementedError()

    @staticmethod
    def get_transforms():
        return None

def get_sets(data_folder, training_augmentation=True):
    """Return hooks to S3DIS dataset train, validation and tests sets.
    """

    train_set = ScanNetDataset(data_folder, 'train', augmentation=training_augmentation)
    valid_set = ScanNetDataset(data_folder, 'valid')
    test_set = ScanNetDataset(data_folder, 'test')

    return train_set, valid_set, test_set

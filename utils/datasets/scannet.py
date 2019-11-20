import torch

class ScanNetDataset(torch.utils.data.Dataset):

    def __init__(self, data_folder, split, path_prefix=None, augmentation=False):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, i):
        raise NotImplementedError()

    @staticmethod
    def get_transforms():
        return None

def get_sets(data_folder, path_prefix=None, training_augmentation=True):
    """Return hooks to S3DIS dataset train, validation and tests sets.
    """

    train_set = ScanNetDataset(data_folder, 'train', path_prefix, augmentation=training_augmentation)
    valid_set = ScanNetDataset(data_folder, 'valid', path_prefix)
    test_set = ScanNetDataset(data_folder, 'test', path_prefix)

    return train_set, valid_set, test_set
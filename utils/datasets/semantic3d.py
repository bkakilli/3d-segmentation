import torch

class Semantic3DDataset(torch.utils.data.Dataset):

    def __init__(self, data_folder="data/semantic3d", split="test", augmentation=False):
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

    train_set = Semantic3DDataset(data_folder, "train", augmentation=training_augmentation)
    valid_set = Semantic3DDataset(data_folder, "val")
    test_set = Semantic3DDataset(data_folder, "test")

    return train_set, valid_set, test_set

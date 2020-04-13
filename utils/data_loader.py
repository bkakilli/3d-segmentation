import os
import sys
import importlib
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from torch.utils.data import DataLoader

def get_loaders(args):

    defined_datasets = ["s3dis", "modelnet", "shapenetparts", "scannet"]
    if args.dataset not in defined_datasets:
        raise ValueError("Undefined dataset: %s"%args.dataset)

    dataset = importlib.import_module('.'+args.dataset, package="datasets")
    train_d, valid_d, test_d = dataset.get_sets(args.dataroot, training_augmentation=(not args.no_augmentation))
    
    # from torch.utils.data import Subset
    # import numpy as np
    # train_d = Subset(train_d, np.arange(10))
    # valid_d = Subset(valid_d, np.arange(10))
    train_l = DataLoader(train_d, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_l = DataLoader(valid_d, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_l  = DataLoader(test_d, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    return train_l, valid_l, test_l

def test():
    """unittest"""
    class DummyArgs(object):
        dataset = "modelnet"
        no_augmentation = False
    
    args = DummyArgs()
    get_loaders(args)

if __name__ == "__main__":
    test()
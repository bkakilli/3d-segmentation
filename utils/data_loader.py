import os
import sys
import importlib
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from torch.utils.data import DataLoader

def get_loaders(args):

    defined_datasets = ["s3dis", "s3dis_cell", "modelnet", "shapenetparts", "scannet", "scannet_rev"]
    if args.dataset not in defined_datasets:
        raise ValueError("Undefined dataset: %s"%args.dataset)

    dataset = importlib.import_module('.'+args.dataset, package="datasets")
    train_d, valid_d, test_d = dataset.get_sets(args.dataroot, split_id=args.split_id, training_augmentation=(not args.no_augmentation))
    
    # from torch.utils.data import Subset
    # import numpy as np
    # train_d = Subset(train_d, np.arange(10))
    # valid_d = Subset(valid_d, np.arange(10))
    print("Loading dataset.")
    train_l = DataLoader(train_d, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=dataset.custom_collate_fn)
    valid_l = DataLoader(valid_d, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=dataset.custom_collate_fn)
    test_l  = DataLoader(test_d, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=dataset.custom_collate_fn)

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
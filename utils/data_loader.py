import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from torch.utils.data import DataLoader

from datasets import s3dis, semantic3d, scannet, modelnet, shapenetparts

def get_loaders(args):
    
    if args.dataset == "s3dis":
        train_d, valid_d, test_d = s3dis.get_sets(args.dataroot, training_augmentation=(not args.no_augmentation))
    elif args.dataset == "semantic3d":
        train_d, valid_d, test_d = semantic3d.get_sets(args.dataroot, training_augmentation=(not args.no_augmentation))
    elif args.dataset == "scannet":
        train_d, valid_d, test_d = scannet.get_sets(args.dataroot, training_augmentation=(not args.no_augmentation))
    elif args.dataset == "modelnet":
        train_d, valid_d, test_d = modelnet.get_sets(args.dataroot, training_augmentation=(not args.no_augmentation))
    elif args.dataset == "shapenetparts":
        train_d, valid_d, test_d = shapenetparts.get_sets(args.dataroot, training_augmentation=(not args.no_augmentation))
    else:
        raise ValueError("Undefined dataset. Valid ones: s3dis, semantic3d, scannet")
    
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
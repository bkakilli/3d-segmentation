import os
import sys
import importlib

CURRENT_FOLDER = os.path.abspath(os.path.dirname(__file__))
sys.path.append(CURRENT_FOLDER)

from torch.utils.data import DataLoader

def get_loaders(args):

    if args.dataset not in [f[:-3] for f in os.listdir(os.path.join(CURRENT_FOLDER, "datasets")) if f.endswith(".py")]:
        raise ValueError("Undefined dataset: %s"%args.dataset)

    dataset = importlib.import_module('.'+args.dataset, package="datasets")
    train_d, valid_d, test_d = dataset.get_sets(args.dataroot, crossval_id=args.crossval_id, training_augmentation=(not args.no_augmentation))
    
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
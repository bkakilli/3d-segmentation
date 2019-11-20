import os
import sys
import shutil
import torch
from datetime import datetime

def persistence(log_dir, model_path, module_name):

    # Initial checkpoint
    checkpoint = {
        "loss": 0,
        "epoch": 0,
        "model_state_dict": None,
        "optimizer_state_dict": None
    }

    # Create log dir if does not exist
    checkpoints_path = os.path.join(log_dir, "checkpoints")
    if not os.path.isdir(log_dir):
        os.makedirs(checkpoints_path)
        shutil.copy(os.path.abspath(__file__), log_dir)
        shutil.copy(os.path.abspath(sys.modules[module_name].__file__), log_dir)
    else:
        if model_path is None:
            ans = input("Folder already exists! Overwrite? [Y/n]: ")
            if not ans in ['y', 'Y', 'yes', 'YES', 'Yes', '']:
                raise FileExistsError("Folder already exists: %s"%(log_dir))
            shutil.copy(os.path.abspath(__file__), log_dir)
            shutil.copy(os.path.abspath(sys.modules[module_name].__file__), log_dir)
            
        else:
            checkpoint = torch.load(model_path)

    return checkpoint

def save_checkpoint(log_dir, container):
    """Save given information (eg. model, optimizer, epoch number etc.) into log_idr
    
    Parameters
    ----------
    log_dir : str
        Path to save
    container : dict
        Information to be saved
    """
    path = os.path.join(log_dir, "checkpoints", "ckpt_%s.pt"%datetime.now().strftime("%Y%m%d_%H%M%S"))
    torch.save(container, path)


def get_lr(optimizer):
    """Return current learining rate of the given optimizer
    
    Parameters
    ----------
    optimizer : Pytorch Optimizer
        Optimizer
    
    Returns
    -------
    float
        learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
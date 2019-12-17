import os
import sys
import json
import requests
import shutil
# import torch
import numpy as np
from datetime import datetime

def persistence(log_dir, model_path, module_name, main_file):

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
        shutil.copy(os.path.abspath(main_file), log_dir)
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

def join_path(*args):
    return os.path.join(*args)

def seed(x):
    np.random.seed(x)
    torch.manual_seed(x)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def slack_message(message, webhook_file="/home/burak/.local/slack_wh.txt"):
    """Sends notification to Slack webhook.
    Webook address should be defined in webhook_file as a single line.
    
    Arguments:
        message {[type]} -- [description]
    """
    with open(webhook_file) as f:
        url = f.readline().strip()
    if isinstance(message, str):
        message = json.dumps({"text": message})
    elif not isinstance(message, dict):
        print("Invalid type as Slack message.")
    
    requests.post(url=url, data=message)


def test():
    slack_message("Test general message.")

if __name__ == "__main__":
    test()
import os
import sys
import json
import requests
import shutil
import torch
import numpy as np
import socket
from datetime import datetime

def move_to(obj, device):
    """Move given object to specified device.
    Object can be dict, list, or tuple.
    """

    if torch.is_tensor(obj):
        return obj.to(device)

    if isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res

    if isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res

    if isinstance(obj, tuple):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return tuple(res)

    raise TypeError("Invalid type for move_to")

def to_tensor(obj):
    """Move given object to tensor.
    Object can be dict, list, tuple, or numpy array.
    """

    if torch.is_tensor(obj):
        return obj

    if isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = to_tensor(v)
        return res

    if isinstance(obj, list):
        res = []
        for v in obj:
            res.append(to_tensor(v))
        return res

    if isinstance(obj, tuple):
        res = []
        for v in obj:
            res.append(to_tensor(v))
        return tuple(res)

    if isinstance(obj, np.ndarray):
        if obj.dtype in [np.float64, np.float32, np.float16, np.int64, np.int32, np.int16, np.int8, np.uint8, np.bool]:
            return torch.from_numpy(obj)
        else:
            return (to_tensor(obj.tolist()))

    raise TypeError("Invalid type for to_tensor")

def persistence(args, module_name, main_file):

    log_dir, model_path = args.logdir, args.model_path
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

    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

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

def gethostname():
    return socket.gethostname()

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

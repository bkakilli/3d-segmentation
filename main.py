from sys import stdout
import time
import argparse

import numpy as np

from scripts.generate_results import get_segmentation_metrics
from models.rle import RLE
from utils import misc, data_loader

import torch
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from svstools.misc import slack_message
from svstools import logger as logging

import torch.autograd.profiler as profiler

def get_arguments():

    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Semantic Segmentation')

    parser.add_argument('--train', action='store_true', help='Trains the model if provided')
    parser.add_argument('--test', action='store_true', help='Evaluates the model if provided')
    parser.add_argument('--dataset', type=str, default='s3dis', help='Experiment dataset')
    parser.add_argument('--root', type=str, default=None, help='Path to data')
    parser.add_argument('--crossval-id', type=int, default=5, help='Split ID to train')
    parser.add_argument('--logdir', type=str, default='log', help='Name of the experiment')
    parser.add_argument('--model-path', type=str, help='Pretrained model path')
    parser.add_argument('--attention', type=str, default='vector', choices=["vector", "scalar"], help='Attention method')
    parser.add_argument('--aggregation', type=str, default='concat', choices=["concat", "sum", "multiply"], help='Attention method')
    parser.add_argument('--local-embedder', type=str, default='pointnet', choices=["dgcnn", "pointnet"], help='Local embedder')
    parser.add_argument('--dim-reduce', action='store_true', help='Apply PCA dimensionality reduction')
    parser.add_argument('--batch-size', type=int, default=1, help='Size of batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of episode to train')
    parser.add_argument('--use-adam', action='store_true', help='Uses Adam optimizer if provided')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.7, help='Learning rate decay rate')
    parser.add_argument('--decay-step', type=float, default=20, help='Learning rate decay step')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0, help='Weight decay rate (L2 regularization)')
    parser.add_argument('--workers', type=int, default=0, help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--print-summary', type=bool,  default=True, help='Whether to print epoch summary')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode. (No status bar, no printout etc.)')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA id. -1 for CPU')
    parser.add_argument('--no-augmentation', action='store_true', help='Disables training augmentation if provided')
    parser.add_argument('--no-parallel', action='store_true', help='Forces to use single GPU if provided')
    parser.add_argument('--webhook', type=str, default='', help='Slack Webhook for notifications')

    return parser.parse_args()


def main():
    # Temporary fix for:
    # RuntimeError: cuDNN error: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in a non-contiguous input.
    # torch.backends.cudnn.enabled = False

    args = get_arguments()
    logging.setup_global_logger("Semantic Segmantation", logpath=misc.join_path(args.logdir, "processing.log"), stdout=(not args.headless))

    # Seed RNG
    misc.seed(args.seed)

    train_loader, valid_loader, test_loader = data_loader.get_loaders(args)

    args.num_classes = train_loader.dataset.num_labels
    model = RLE(**vars(args))
    if torch.cuda.device_count() > 1 and not args.no_parallel:
        model = torch.nn.DataParallel(model)

    if args.train:
        train(model, train_loader, valid_loader, args)

    if args.test:
        test(model, test_loader, args)


def run_one_epoch(model, tqdm_iterator, mode, get_locals=False, optimizer=None, loss_update_interval=1000):
    """Definition of one epoch procedure.
    """
    if mode == "train":
        assert optimizer is not None
        model.train()
    else:
        model.eval()
        # Following is a hack for CUDA OF OF MEMORY issue when backward is not called.
        # Reverse is done at the end of the function
        param_grads = []
        for param in model.parameters():
            param_grads += [param.requires_grad]
            param.requires_grad = False

    summary = {"losses": [], "logits": [], "labels": []}

    device = next(model.parameters()).device
    loss_fcn = model.module.get_loss if isinstance(model, torch.nn.DataParallel) else model.get_loss

    for i, batch_cpu in enumerate(tqdm_iterator):
        # X, groups, y = X_cpu.to(device), G_cpu.to(device), y_cpu.to(device)
        batch = misc.move_to(batch_cpu, device)
        X, y = batch
        X_cpu, y_cpu = batch_cpu

        if optimizer:
            optimizer.zero_grad()

        logits = model(X)
        loss = loss_fcn(logits, y)
        summary["losses"] += [loss.item()]

        # np.savez("/seg/scripts/output.npz", cloud=X_cpu.numpy(), labels=y_cpu.numpy(), preds=prod_logits.cpu().detach().numpy())
        # asdasd

        if mode == "train":

            # Apply back-prop
            loss.backward()
            optimizer.step()

            # Display
            if loss_update_interval > 0 and i%loss_update_interval == 0:
                tqdm_iterator.set_description("Loss: %.3f" % (np.mean(summary["losses"])))

        if get_locals:
            summary["logits"] += [logits.cpu().detach().numpy()]
            summary["labels"] += [y_cpu.numpy()]

        # torch.cuda.empty_cache()

    # Following is the reverse of the hack defined above
    if mode != "train":
        for param, value in zip(model.parameters(), param_grads):
            param.requires_grad = value

    # if get_locals:
        # summary["logits"] = np.concatenate(summary["logits"], axis=0)
        # summary["labels"] = np.concatenate(summary["labels"], axis=0)

    return summary

def test(model, test_loader, args):

    # Set device
    assert args.cuda < 0 or torch.cuda.is_available()
    device_tag = "cpu" if args.cuda == -1 else "cuda:%d"%args.cuda
    device = torch.device(device_tag)

    # Set model and loss
    model = model.to(device)

    # Get current state
    if args.model_path is not None:
        state = torch.load(args.model_path)
        model.load_state_dict(state["model_state_dict"])
        logging.info("Loaded pre-trained model from %s"%args.model_path)

    def test_one_epoch():
        iterations = tqdm(test_loader, unit='batch', desc="Testing", disable=args.headless)
        ep_sum = run_one_epoch(model, iterations, "test", get_locals=True, loss_update_interval=-1)

        preds = ep_sum["logits"].argmax(axis=-2)
        summary = get_segmentation_metrics(ep_sum["labels"], preds)
        summary["Loss/test"] = np.mean(ep_sum["losses"])

        np.savez_compressed("output.npz", summary=summary, labels=ep_sum["labels"], preds=preds)

        return summary

    summary = test_one_epoch()
    summary["IoU per Class"] = np.array2string(np.array(summary["IoU per Class"]), 1000, 3, False)
    summary_string = misc.json.dumps(summary, indent=2)
    logging.info("Testing summary:\n%s" % (summary_string))


    if args.webhook != '':
        slack_message("Test result in %s:\n%s" % (args.logdir, summary_string), url=args.webhook)


def train(model, train_loader, valid_loader, args):
    """Trainer function for PointNet
    """
    if args.webhook != '':
        slack_message("Training started in %s (%s)" % (args.logdir, misc.gethostname()), url=args.webhook)
        

    # Set device
    assert args.cuda < 0 or torch.cuda.is_available()
    device_tag = "cpu" if args.cuda == -1 else "cuda:%d"%args.cuda
    device = torch.device(device_tag)

    # Set model and label weights
    model = model.to(device)
    model.labelweights = torch.tensor(train_loader.dataset.labelweights, device=device, requires_grad=False)

    # Set optimizer (default SGD with momentum)
    if args.use_adam:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)

    # Get current state
    model_object = model.module if isinstance(model, torch.nn.DataParallel) else model
    module_file = misc.sys.modules[model_object.__class__.__module__].__file__
    state = misc.persistence(args, module_file=module_file, main_file=__file__)
    init_epoch = state["epoch"]

    if state["model_state_dict"]:
        logging.info("Loading pre-trained model from %s"%args.model_path)
        model.load_state_dict(state["model_state_dict"])

    if state["optimizer_state_dict"]:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                  milestones=np.arange(
                                                      args.decay_step,
                                                      args.epochs,
                                                      args.decay_step).tolist(),
                                                  gamma=args.lr_decay,
                                                  last_epoch=init_epoch-1)


    def train_one_epoch():
        iterations = tqdm(train_loader, unit='batch', leave=False, disable=args.headless)
        ep_sum = run_one_epoch(model, iterations, "train", optimizer=optimizer, loss_update_interval=10)

        summary = {"Loss": np.mean(ep_sum["losses"])}
        return summary

    def eval_one_epoch():
        iterations = tqdm(valid_loader, unit='batch', leave=False, desc="Validation", disable=args.headless)
        ep_sum = run_one_epoch(model, iterations, "test", get_locals=True, loss_update_interval=-1)

        # preds = ep_sum["logits"].argmax(axis=-2)
        preds = [l[0].argmax(axis=0) for l in ep_sum["logits"]]
        labels = [l[0] for l in ep_sum["labels"]]
        summary = get_segmentation_metrics(labels, preds)
        summary["Loss"] = float(np.mean(ep_sum["losses"]))
        return summary

    # Train for multiple epochs
    tensorboard = SummaryWriter(log_dir=misc.join_path(args.logdir, "logs"))
    tqdm_epochs = tqdm(range(init_epoch, args.epochs), total=args.epochs, initial=init_epoch, unit='epoch', desc="Progress", disable=args.headless)
    logging.info("Training started.")
    for e in tqdm_epochs:
        train_summary = train_one_epoch()
        valid_summary = eval_one_epoch()
        # valid_summary={"Loss/validation":0}
        train_summary["Learning Rate"] = lr_scheduler.get_last_lr()[-1]

        train_summary = {f"Train/{k}": train_summary.pop(k) for k in list(train_summary.keys())}
        valid_summary = {f"Validation/{k}": valid_summary.pop(k) for k in list(valid_summary.keys())}
        summary = {**train_summary, **valid_summary}

        if args.print_summary:
            tqdm_epochs.clear()
            logging.info("Epoch %d summary:\n%s\n" % (e+1, misc.json.dumps((summary), indent=2)))

        # Update learning rate and save checkpoint
        lr_scheduler.step()
        misc.save_checkpoint(args.logdir, {
            "epoch": e+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": summary["Validation/Loss"],
            "summary": summary
        })

        # Write summary
        for name, val in summary.items():
            if "IoU per Class" in name: continue
            tensorboard.add_scalar(name, val, global_step=e+1)

    if args.webhook != '':
        slack_message("Training finished in %s (%s)" % (args.logdir, misc.gethostname()), url=args.webhook)

if __name__ == "__main__":
    main()
